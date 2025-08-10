import math
import monai
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import CLIPModel
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv, LlamaRMSNorm
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.utils import logging

from Config import LLAMATokenizerPath, ClipPath, ClipGenFeatureSize, ClipGenFeaturePath
from ModelMaskDecoder import VisionDecoder
from ModelGlobalDecoder import GlobalDecoder

torch.set_printoptions(linewidth=1000)

logger = logging.get_logger(__name__)

def custom_llama_attention_forward_eager(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    custom_stuffs = None,
    last_layer_ts_embedding=None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn("Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`")

    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        assert False, "We are supposed to set config.pretraining_tp to 1."
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        # Here we do something of ourself
        # In custom_stuffs, we have 'custom_current_layer', 'custom_project_module', 'custom_qkv_dimension', 'custom_active_rate', 'ts_ema_alpha', 'last_layer_ts_embedding'
        ts_embedding = self.custom_projection_to_ts(hidden_states)
        if last_layer_ts_embedding is None:
            pass
        else:
            ts_embedding = (1-custom_stuffs['ts_ema_alpha'])*ts_embedding + custom_stuffs['ts_ema_alpha']*last_layer_ts_embedding
        ts_embedding = self.custom_norm(ts_embedding)
        if not custom_stuffs['inference']:
            sampled_custom_active_rate = torch.normal(mean=torch.tensor([custom_stuffs['custom_active_rate']]), std=custom_stuffs['custom_active_rate']*0.1)[0]
            # sampled_custom_active_rate = custom_stuffs['custom_active_rate']
            kth_largerst_values, kth_largest_indices = torch.kthvalue(ts_embedding, k=((1-sampled_custom_active_rate)*ts_embedding.shape[-1]).long(), dim=-1, keepdim=True)
            ts_embedding_filtered = torch.nn.functional.relu(ts_embedding - kth_largerst_values)
        else:
            ts_embedding_filtered = ts_embedding
        last_layer_ts_embedding = ts_embedding # Here we use the unfiltered embedding for the next layer

        origin_embedding = self.custom_dropout(self.custom_projection_to_origin(ts_embedding_filtered))
        split_origin_embedding = torch.split(origin_embedding, custom_stuffs['custom_qkv_dimension'], dim=-1)

        if 'qv' == custom_stuffs['custom_project_module']:
            assert len(split_origin_embedding) == 2
            q_from_ts, k_from_ts, v_from_ts = split_origin_embedding[0], 0, split_origin_embedding[1]
        elif 'kv' == custom_stuffs['custom_project_module']:
            assert len(split_origin_embedding) == 2
            q_from_ts, k_from_ts, v_from_ts = 0, split_origin_embedding[0], split_origin_embedding[1]
        elif 'qkv' == custom_stuffs['custom_project_module']:
            assert len(split_origin_embedding) == 3
            q_from_ts, k_from_ts, v_from_ts = split_origin_embedding[0], split_origin_embedding[1], split_origin_embedding[1]
        else:
            raise NotImplementedError
        query_states = self.q_proj(hidden_states) + q_from_ts
        key_states = self.k_proj(hidden_states) + k_from_ts
        value_states = self.v_proj(hidden_states) + v_from_ts

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value, last_layer_ts_embedding

def custom_llama_attention_forward_flash(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    custom_stuffs = None,
    last_layer_ts_embedding=None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop("padding_mask")

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()
    
    # Here we do something of ourself
    # In custom_stuffs, we have 'custom_current_layer', 'custom_project_module', 'custom_qkv_dimension', 'custom_active_rate', 'ts_ema_alpha', 'last_layer_ts_embedding'
    ts_embedding = self.custom_projection_to_ts(hidden_states)
    if last_layer_ts_embedding is None:
        pass
    else:
        ts_embedding = (1-custom_stuffs['ts_ema_alpha'])*ts_embedding + custom_stuffs['ts_ema_alpha']*last_layer_ts_embedding
    ts_embedding = self.custom_norm(ts_embedding)
    if not custom_stuffs['inference']:
        sampled_custom_active_rate = torch.normal(mean=torch.tensor([custom_stuffs['custom_active_rate']]), std=custom_stuffs['custom_active_rate']*0.1)[0]
        kth_largerst_values, kth_largest_indices = torch.kthvalue(ts_embedding, k=((1-sampled_custom_active_rate)*ts_embedding.shape[-1]).long(), dim=-1, keepdim=True)
        ts_embedding_filtered = torch.nn.functional.relu(ts_embedding - kth_largerst_values)
    else:
        ts_embedding_filtered = ts_embedding
    last_layer_ts_embedding = ts_embedding # Here we use the unfiltered embedding for the next layer

    origin_embedding = self.custom_dropout(self.custom_projection_to_origin(ts_embedding_filtered))
    split_origin_embedding = torch.split(origin_embedding, custom_stuffs['custom_qkv_dimension'], dim=-1)

    if 'qv' == custom_stuffs['custom_project_module']:
        assert len(split_origin_embedding) == 2
        q_from_ts, k_from_ts, v_from_ts = split_origin_embedding[0], 0, split_origin_embedding[1]
    elif 'kv' == custom_stuffs['custom_project_module']:
        assert len(split_origin_embedding) == 2
        q_from_ts, k_from_ts, v_from_ts = 0, split_origin_embedding[0], split_origin_embedding[1]
    elif 'qkv' == custom_stuffs['custom_project_module']:
        assert len(split_origin_embedding) == 3
        q_from_ts, k_from_ts, v_from_ts = split_origin_embedding[0], split_origin_embedding[1], split_origin_embedding[1]
    else:
        raise NotImplementedError
    query_states = self.q_proj(hidden_states) + q_from_ts
    key_states = self.k_proj(hidden_states) + k_from_ts
    value_states = self.v_proj(hidden_states) + v_from_ts

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = self._flash_attention_forward(
        query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value, last_layer_ts_embedding

def custom_llama_layer_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        custom_stuffs = None,
        last_layer_ts_embedding=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value, last_layer_ts_embedding = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            custom_stuffs=custom_stuffs,
            last_layer_ts_embedding=last_layer_ts_embedding,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        outputs += (last_layer_ts_embedding,)
        return outputs

def custom_llama_model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        custom_stuffs = None,
        last_layer_ts_embedding=None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None


        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Define some custom things
            if custom_stuffs['custom_current_layer'] is not None:
                custom_stuffs['custom_current_layer'] = custom_stuffs['custom_current_layer'] + 1
            else:
                custom_stuffs['custom_current_layer'] = 0
            if self.gradient_checkpointing and self.training:
                if i < custom_stuffs['ts_start_layer']:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                    )
                else:                    
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        custom_stuffs,
                        last_layer_ts_embedding,
                    )
                    last_layer_ts_embedding = layer_outputs[-1]
                    
            else:
                if i < custom_stuffs['ts_start_layer']:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        custom_stuffs=custom_stuffs,
                        last_layer_ts_embedding=last_layer_ts_embedding,
                    )
                    last_layer_ts_embedding = layer_outputs[-1]

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                
        custom_last_hidden_state = hidden_states
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ), custom_last_hidden_state

def custom_llama_for_casuallm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    custom_stuffs = None,
    last_layer_ts_embedding=None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs, custom_last_hidden_state = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        custom_stuffs=custom_stuffs,
        last_layer_ts_embedding=last_layer_ts_embedding,
    )

    hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    ), custom_last_hidden_state

class Model(torch.nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.image_size = opt.image_size
        self.use_clip_project = opt.use_clip_project
        self.pad_id = opt.pad_id
        self.glb_token_id = opt.glb_token_idx
        self.lcl_token_ids = opt.lcl_token_idxs
        self.vocab_size = opt.vocab_size
        self.vision_token_length = opt.vision_token_length
        self.attention = opt.attention # ['eager', 'flash_attention_2']
        self.checkpoint = opt.checkpoint

        self.ts_start_layer = opt.ts_start_layer
        self.space_project_modules = opt.space_project_modules # qv, kv, qkv
        assert self.space_project_modules in ['qv', 'kv', 'qkv']
        self.number_project_modules = len(self.space_project_modules)
        self.ts_space_dimension = opt.ts_space_dimension
        self.ts_space_active_rate = opt.ts_space_active_rate
        self.ts_space_dropout = opt.ts_space_dropout
        self.ts_ema_alpha = opt.ts_ema_alpha
        
        self.decoder_extract_layers = opt.decoder_extract_layers # e.g., [9, 6, 3], global and local share
        self.ldecoder_reduce_dim = opt.ldecoder_reduce_dim
        self.ldecoder_num_attention_heads = opt.ldecoder_num_attention_heads
        self.ldecoder_dropout = opt.ldecoder_dropout
        self.ldecoder_intermediate_size = opt.ldecoder_intermediate_size
        
        self.gdecoder_target_size = ClipGenFeatureSize
        self.gdecoder_hidden_dim = opt.gdecoder_hidden_dim
        self.gdecoder_hidden_dropout = opt.gdecoder_hidden_dropout

        self.a1, self.a2, self.a3 = opt.a1, opt.a2, opt.a3
        self.c1, self.c2 = opt.c1, opt.c2
        
        # Define the text foundation model
        self.text_dtype = torch.float16 if opt.float16 else torch.bfloat16
        self.text_model_config = LlamaConfig.from_pretrained(pretrained_model_name_or_path=LLAMATokenizerPath, 
                                pad_token_id=self.pad_id, rms_norm_eps=1e-6, attn_implementation=self.attention)
        self.text_model = LlamaForCausalLM.from_pretrained(LLAMATokenizerPath, config=self.text_model_config, 
                                load_in_8bit=False, torch_dtype=self.text_dtype)
        self.text_model.resize_token_embeddings(self.vocab_size)

        self.text_feature_size = self.text_model_config.hidden_size # Size of text features 4096 for 7B and 5120 for 13B
        self.text_inter_feature_size = self.text_model_config.intermediate_size # Intermediate size in MLP, 11008 for 7B and 13824 for 13B
        self.text_num_heads = self.text_model_config.num_key_value_heads # Number of heads, 32 for 7B and 40 for 13B
        self.text_dimension_heads = self.text_feature_size // self.text_num_heads 
        self.text_hidden_layers = self.text_model_config.num_hidden_layers # Number of layers, 32 for 7B and 40 for 13B

        # Replace the forward function and define some custom stuffs
        new_llama_for_casuallm_forward = custom_llama_for_casuallm_forward.__get__(self.text_model, self.text_model.__class__)
        setattr(self.text_model, 'forward', new_llama_for_casuallm_forward)
        new_llama_model_forward = custom_llama_model_forward.__get__(self.text_model.model, self.text_model.model.__class__)
        setattr(self.text_model.model, 'forward', new_llama_model_forward)
        for i in range(len(self.text_model.model.layers)):
            if i < self.ts_start_layer:
                continue
            new_llama_layer_forward = custom_llama_layer_forward.__get__(self.text_model.model.layers[i], self.text_model.model.layers[i].__class__)
            setattr(self.text_model.model.layers[i], 'forward', new_llama_layer_forward)
            
            if self.attention == 'eager':
                new_llama_attention_forward = custom_llama_attention_forward_eager.__get__(self.text_model.model.layers[i].self_attn, self.text_model.model.layers[i].self_attn.__class__)
                setattr(self.text_model.model.layers[i].self_attn, 'forward', new_llama_attention_forward)
            elif self.attention == 'flash_attention_2':
                new_llama_attention_forward = custom_llama_attention_forward_flash.__get__(self.text_model.model.layers[i].self_attn, self.text_model.model.layers[i].self_attn.__class__)
                setattr(self.text_model.model.layers[i].self_attn, 'forward', new_llama_attention_forward)
            else:
                raise NotImplementedError
                
            self.text_model.model.layers[i].self_attn.custom_projection_to_ts = torch.nn.Linear(self.text_feature_size, self.ts_space_dimension, bias=False)
            self.text_model.model.layers[i].self_attn.custom_norm = LlamaRMSNorm(self.ts_space_dimension, eps=self.text_model_config.rms_norm_eps)
            self.text_model.model.layers[i].self_attn.custom_dropout = torch.nn.Dropout(self.ts_space_dropout)
            self.text_model.model.layers[i].self_attn.custom_projection_to_origin = torch.nn.Linear(self.ts_space_dimension, self.text_num_heads*self.text_dimension_heads*self.number_project_modules, bias=False)
        if self.checkpoint:
            self.text_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})

        # Define the vision foundation model
        self.vision_dtype = self.text_dtype if opt.vision_dtype_half else torch.float32
        vision_model = CLIPModel.from_pretrained(ClipPath)
        self.vision_encoder = vision_model.vision_model
        self.vision_projection = vision_model.visual_projection if self.use_clip_project else None
        
        self.vision_feature_size = vision_model.vision_embed_dim # Dimension of vision features, 768/1024/1024
        self.vision_projected_size = vision_model.projection_dim # Dits_ema_alphamension after the image_encoder_projection, 512/768/768
        self.vision_encoder_patch_size = vision_model.vision_model.config.patch_size # The patch size of clip encoder
        self.vision_encoder_layers = vision_model.vision_model.config.num_hidden_layers
        
        if not self.use_clip_project: # the local and global image features are transformed at the same time
            self.fc_vision_to_text = torch.nn.Linear(self.vision_feature_size, self.text_feature_size, bias=False)
            # torch.nn.init.constant_(self.fc_vision_to_text.weight, 1.0)
            # torch.nn.init.constant_(self.fc_vision_to_text.bias, 1.0)
        else: # the local and global image features are transformed seperately
            self.fc_vision_to_text = torch.nn.Linear(self.vision_feature_size, self.text_feature_size, bias=False)
            self.fc_vision_global_to_text = torch.nn.Linear(self.vision_projected_size, self.text_feature_size, bias=False)
            # torch.nn.init.constant_(self.fc_vision_to_text.weight, 1.0)
            # torch.nn.init.constant_(self.fc_vision_to_text.bias, 1.0)

        # Define the decoder for generation
        self.decoder_local = VisionDecoder(text_feature_size=self.text_feature_size, vision_feature_channel=self.vision_feature_size, num_out_channel=2, 
                                           extract_layers=self.decoder_extract_layers, vision_encoder_layers=self.vision_encoder_layers, patch_size=self.vision_encoder_patch_size, 
                                           decoder_reduce_dim=self.ldecoder_reduce_dim, decoder_num_attention_heads=self.ldecoder_num_attention_heads, 
                                           decoder_dropout=self.ldecoder_dropout, decoder_intermediate_size=self.ldecoder_intermediate_size)
        # self.decoder_vision_hub = VisionHub(text_size=self.text_feature_size, vae_size=self.vae_feature_size, vision_hub_hidden_dim=self.vision_hub_hidden_dim, vision_hub_dropout=self.vision_hub_dropout)
        self.decoder_global = GlobalDecoder(self.text_feature_size, self.gdecoder_target_size, self.gdecoder_hidden_dim, self.gdecoder_hidden_dropout)
        
        # Define some loss functions
        self.text_generation_loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.global_generation_loss_function = torch.nn.MSELoss()
        self.local_generation_loss_function1 = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.9, 1.1]).to(self.opt.device))
        self.local_generation_loss_function2 = monai.losses.DiceLoss(reduction='mean', squared_pred=True, softmax=True)
                

    # images: [bs, 3, h, w]  text: [bs, len]
    # tasks in [0, 1, 2] for caption, generation, and complex_segment
    def forward(self, image_pixel_values, text_input_ids, text_attention_mask, label_input_ids, global_labels, local_labels, tasks, inference=False):
        # Extract vision features
        with torch.no_grad():
            with torch.autocast('cuda', dtype=self.vision_dtype):
                vision_outputs = self.vision_encoder(pixel_values=image_pixel_values, output_hidden_states=True)
                vision_last_hidden_state = vision_outputs['last_hidden_state'].detach()
                vision_pooled_output = vision_outputs['pooler_output'].detach()
                vision_hidden_states = vision_outputs['hidden_states'][1:] # 13/25 layers in all, 0th is the embedding
                vision_hidden_states = [vision_hidden_state.detach() for vision_hidden_state in vision_hidden_states]

        if not self.use_clip_project: # the local and global image features are transformed at the same time
            transformed_vision_last_hidden_state = self.fc_vision_to_text(vision_last_hidden_state)
            # We put the global token into the last position
        else: # the local and global image features are transformed seperately
            transformed_vision_last_hidden_state = self.fc_vision_to_text(vision_last_hidden_state[:, 1:, :])
            vision_pooled_output = self.vision_projection(vision_pooled_output)
            transformed_vision_pooled_output = self.fc_vision_global_to_text(vision_pooled_output)
            transformed_vision_last_hidden_state = torch.cat((transformed_vision_pooled_output.unsqueeze(1), transformed_vision_last_hidden_state), dim=1)

        # Extract text features and interact with vision features
        custom_stuffs = {
            'ts_start_layer': self.ts_start_layer,
            'custom_current_layer': None,
            'custom_project_module': self.space_project_modules,
            'custom_qkv_dimension': self.text_num_heads*self.text_dimension_heads,
            'custom_active_rate': self.ts_space_active_rate,
            'ts_space_active_rate': self.ts_space_active_rate,
            'ts_ema_alpha': self.ts_ema_alpha,
            'inference': inference, 
        }
        with torch.autocast('cuda', dtype=self.text_dtype):
            # Here, we do the operation based on some assumption:
            # 1. generation task sentences are always shorter than others
            # 2. each task have the sample in this batch
            # 3. the padding is right-padding
            text_embed = self.text_model.model.embed_tokens(text_input_ids)
            hybrid_text_embed, hybrid_attention_mask, hybrid_labels = self.reconstruct_inputs_and_labels(text_embed, transformed_vision_last_hidden_state, 
                                                                                                        text_attention_mask, label_input_ids, tasks)
            # hybrid_text_embed, hybrid_attention_mask, hybrid_labels = self.reconstruct_inputs_and_labels_all_image(text_embed, transformed_vision_last_hidden_state, 
            #                                                                                             text_attention_mask, label_input_ids, tasks)
            
            # Pass the ts_embedding as a parameter as checkpointing cannot recognize the tensors in List or Dict, etc
            text_outputs, custom_last_hidden_state = self.text_model(input_ids=None, inputs_embeds=hybrid_text_embed, attention_mask=hybrid_attention_mask, 
                                           past_key_values=None, labels=None, output_hidden_states=True, custom_stuffs=custom_stuffs, 
                                           last_layer_ts_embedding=None, use_cache=False)
        text_prediction = text_outputs['logits']
        text_hidden_states = text_outputs['hidden_states'] # layer * tensor
        last_text_hidden_states = text_hidden_states[-1] # The hidden states of the last layer, which is used for the prediction in LLM
        # last_text_hidden_states = custom_last_hidden_state

        # Predict the global and local results
        shifted_hybrid_labels = torch.cat([hybrid_labels[:, 1:], (torch.ones(hybrid_labels.shape[0], 1)*(-100)).long().to(hybrid_labels.device)], dim=1)
        glb_token_indices = (shifted_hybrid_labels.detach()==self.glb_token_id)
        lcl_token_indices = shifted_hybrid_labels.detach().cpu()
        lcl_token_indices.apply_(lambda x: x in self.lcl_token_ids)
        lcl_token_indices = lcl_token_indices.to(self.opt.device).bool()
        glb_tokens_hidded_states = last_text_hidden_states[glb_token_indices]
        lcl_tokens_hidded_states = last_text_hidden_states[lcl_token_indices]
        lcl_token_indices_count = lcl_token_indices.sum(-1)
        
        glb_decoder_prediction = self.decoder_global(glb_tokens_hidded_states)
        if 'UN' in ClipGenFeaturePath: # Unnormlized case
            glb_decoder_prediction = glb_decoder_prediction
        else:  # Normlized case
            glb_decoder_prediction = glb_decoder_prediction / glb_decoder_prediction.norm(dim=-1, keepdim=True)
        
        # Here we have the assumption:
        # 1. Each sample only have 0 or 1 <GLB> token
        local_extracted_vision_hidden_states = [torch.repeat_interleave(vision_hidden_states[i], lcl_token_indices_count, dim=0) for i in self.decoder_extract_layers]
        lcl_token_generated_results = self.decoder_local(local_extracted_vision_hidden_states, lcl_tokens_hidded_states)[0]
        lcl_token_generated_results = F.interpolate(lcl_token_generated_results, [self.image_size, self.image_size], mode='bilinear', align_corners=True)
        

        # Compute corresponding loss
        text_generation_loss = self.compute_text_generation_loss(text_prediction, hybrid_labels)
        local_generation_loss = self.compute_local_generation_loss(lcl_token_generated_results, local_labels)
        global_generation_loss = self.global_generation_loss_function(glb_decoder_prediction, global_labels)
        final_loss = self.a1 * text_generation_loss + self.a2 * global_generation_loss + self.a3 * local_generation_loss
        
        
        if 'UN' in ClipGenFeaturePath:
            glb_decoder_prediction_normal = glb_decoder_prediction / glb_decoder_prediction.norm(dim=-1, keepdim=True)
            global_labels_normal = global_labels / global_labels.norm(dim=-1, keepdim=True)
            global_generation_mae = torch.abs(glb_decoder_prediction_normal - global_labels_normal).mean()
        else:
            global_generation_mae = torch.abs(glb_decoder_prediction - global_labels).mean()
        
        return {
            'text_prediction': text_prediction, 
            # 'text_past_key_values': text_past_key_values, 
            # 'hybrid_labels': hybrid_labels, 
            # 'last_text_hidden_states': last_text_hidden_states,
            'lcl_token_generated_results': lcl_token_generated_results,
            'text_generation_loss': text_generation_loss,
            'global_generation_loss': global_generation_loss,
            'global_generation_mae': global_generation_mae.detach().cpu().item(),
            'local_generation_loss': local_generation_loss,
            'final_loss': final_loss,
        }
        
    def reconstruct_inputs_and_labels(self, text_embed, vision_embed, text_attention_mask, label_input_ids, tasks):
        tasks = torch.tensor(tasks)
        is_generation_task = (tasks==1)
        not_generation_task = (tasks!=1)
        is_generation_task_count = is_generation_task.sum().cpu().item()
        not_generation_task_count = not_generation_task.sum().cpu().item()
        
        batch_size = text_embed.shape[0]
        text_length = text_embed.shape[1]
        vision_length = vision_embed.shape[1]
        assert self.vision_token_length == vision_length

        hybrid_text_embed = torch.zeros((batch_size, vision_length+text_length, self.text_feature_size), device=label_input_ids.device)
        hybrid_text_embed[not_generation_task] = torch.cat((vision_embed[not_generation_task, :, :], text_embed[not_generation_task, :, :]), dim=1)
        hybrid_text_embed[is_generation_task] = torch.cat(
            (
                text_embed[is_generation_task, :, :].float(), 
                # torch.zeros(is_generation_task_count, vision_length, self.text_feature_size, device=label_input_ids.device)
                self.text_model.model.embed_tokens.weight[0].unsqueeze(0).unsqueeze(0).repeat(is_generation_task_count, vision_length, 1).float(),
                # torch.full((is_generation_task_count, vision_length, self.text_feature_size), self.text_model.model.embed_tokens.weight[0], device=label_input_ids.device)
            ), dim=1)

        hybrid_attention_mask = torch.ones((batch_size, vision_length+text_length), device=label_input_ids.device)*3
        hybrid_attention_mask[not_generation_task] = torch.cat(
            (torch.ones((not_generation_task_count, vision_length), device=text_attention_mask.device), text_attention_mask[not_generation_task, :]),
            dim=1)
        hybrid_attention_mask[is_generation_task] = torch.cat((
            text_attention_mask[is_generation_task, :], 
            torch.zeros((is_generation_task_count, vision_length), device=label_input_ids.device)), dim=1)
        hybrid_attention_mask = hybrid_attention_mask.long()

        hybrid_labels = (torch.ones((batch_size, vision_length+text_length), device=label_input_ids.device) * 999999).long()
        hybrid_labels[not_generation_task] = torch.cat(
            ((torch.ones((not_generation_task_count, vision_length), device=label_input_ids.device) * -100).long(), label_input_ids[not_generation_task, :]), 
            dim=1)
        hybrid_labels[is_generation_task] = torch.cat(
            (label_input_ids[is_generation_task, :], (torch.ones((is_generation_task_count, vision_length), device=label_input_ids.device) * -100).long()), 
            dim=1)
        
        return hybrid_text_embed, hybrid_attention_mask, hybrid_labels
    
    def reconstruct_inputs_and_labels_all_image(self, text_embed, vision_embed, text_attention_mask, label_input_ids, tasks):        
        batch_size = text_embed.shape[0]
        text_length = text_embed.shape[1]
        vision_length = vision_embed.shape[1]
        assert self.vision_token_length == vision_length

        hybrid_text_embed = torch.cat((vision_embed, text_embed), dim=1)
        hybrid_attention_mask = torch.cat((torch.ones((batch_size, vision_length), device=text_attention_mask.device), text_attention_mask), dim=1).long()
        hybrid_labels = torch.cat(((torch.ones((batch_size, vision_length), device=label_input_ids.device) * -100), label_input_ids), dim=1).long()
        
        return hybrid_text_embed, hybrid_attention_mask, hybrid_labels

    def compute_text_generation_loss(self, text_predictions, labels):
        shift_logits = text_predictions[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        
        shift_labels = shift_labels.to(shift_logits.device)
        text_loss = self.text_generation_loss_function(shift_logits, shift_labels)

        return text_loss

    def compute_local_generation_loss(self, local_predictions, labels):
        bs, channel, height, width = local_predictions.shape
        
        local_loss1 = self.local_generation_loss_function1(local_predictions.reshape(bs*height*width, channel), labels.reshape(bs*height*width,))
        labels = monai.losses.dice.one_hot(labels[:, None, ...], num_classes=2)
        local_loss2 = self.local_generation_loss_function2(local_predictions, labels)
        local_loss = self.c1 * local_loss1 + self.c2 * local_loss2
        return local_loss

