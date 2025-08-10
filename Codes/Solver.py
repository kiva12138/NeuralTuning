import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup
from transformers import LlamaTokenizer
import torch.distributed as dist
import bitsandbytes

from Dataset import get_data_loader_ddp, get_complete_input_text
from ModelEMA import Model
from Utils import compute_IoU_2class_batch, compute_dice_from_iou, log_message, set_logger
from Config import LLAMATokenizerPath, ClipPath, ClipTokens


class Solver():
    def __init__(self, opt):
        self.opt = opt
        
        self.task_path, self.writer, self.best_valid_model_path, self.best_test_model_path, self.newest_model_path = self.prepare_checkpoint_log()
        log_message("Making logger and dataset...", self.opt.local_rank)
        log_message(str(self.opt), self.opt.local_rank)
        self.data_loader_train, self.data_loader_val, self.data_loader_test = get_data_loader_ddp(self.opt.batch_size, self.opt.drop_last, 
                                                                                                  self.opt.num_workers, self.opt.persistent_workers, self.opt.pin_memory, 
                                                                                                  max_length=self.opt.max_length, mask_length=self.opt.mask_length)
        log_message("Number of training steps in each epoch(train): {}".format(len(self.data_loader_train)), self.opt.local_rank)
        log_message("Number of training steps in each epoch(valid): {}".format(len(self.data_loader_val)), self.opt.local_rank)
        log_message("Number of training steps in each epoch(test): {}".format(len(self.data_loader_test)), self.opt.local_rank)
        
        log_message("Making model and optimizer...", self.opt.local_rank)
        tokenizer_question = LlamaTokenizer.from_pretrained(LLAMATokenizerPath, add_bos_token=True, add_eos_token=False, padding_side='right')
        tokenizer_answer = LlamaTokenizer.from_pretrained(LLAMATokenizerPath, add_bos_token=False, add_eos_token=True, padding_side='right')
        tokenizer_question.pad_token = tokenizer_question.unk_token # Follow previous works, we set pad token to unk token
        tokenizer_answer.pad_token = tokenizer_answer.unk_token
        new_tokens = ["<GLB>"]
        for i in range(1, self.opt.mask_length+1):
            new_tokens.append('<LCL-{}>'.format(i))
            
        tokenizer_question.add_tokens(new_tokens) # Add two external tokens
        tokenizer_answer.add_tokens(new_tokens) # Add two external tokens
        self.opt.tokenizer_question = tokenizer_question
        self.opt.tokenizer_answer = tokenizer_answer
        
        glb_token_idx = tokenizer_question("<GLB>", add_special_tokens=False)['input_ids'][-1] # [_, <GLB>], [_, <LCL>]
        assert glb_token_idx >= len(tokenizer_question)-len(new_tokens)
        lcl_token_idxs = []
        for i in range(1, self.opt.mask_length+1):
            lcl_token_idx = tokenizer_question("<LCL-{}>".format(i), add_special_tokens=False)['input_ids'][-1]
            lcl_token_idxs.append(lcl_token_idx)
            assert lcl_token_idx >= len(tokenizer_question)-len(new_tokens)
        
        self.opt.glb_token_idx, self.opt.lcl_token_idxs = glb_token_idx, lcl_token_idxs
        self.opt.pad_id = tokenizer_question.pad_token_id
        self.opt.vocab_size = len(tokenizer_question)
        self.opt.vision_token_length = ClipTokens[ClipPath]
        
        log_message('All number of tokens: {}'.format(len(tokenizer_question)), self.opt.local_rank) # 32002
        
        self.model = Model(self.opt).to(opt.device)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.freeze_params()
        self.model = DDP(self.model, device_ids=[opt.gpu_id], find_unused_parameters=False)
        
        self.optimizer, self.lr_schedule = self.get_optimizer(self.model)
        # self.scaler = torch.cuda.amp.GradScaler()

    def solve(self):
        log_message("Start training...", self.opt.local_rank)
        best_valid_model_states, best_test_model_states = dict(), dict()
        best_valid_score, best_test_score = None, None

        for epoch in range(self.opt.epochs_num):
            self.data_loader_train.sampler.set_epoch(epoch)
            self.data_loader_val.sampler.set_epoch(epoch)
            self.data_loader_test.sampler.set_epoch(epoch)
            
            train_loss, train_score = self.train(self.data_loader_train)
            val_loss, val_score = self.evaluate(self.data_loader_val)
            test_loss, test_score = self.evaluate(self.data_loader_test)
            
            if self.opt.warmup_steps <= 0: # If using warming up, we use batch-based lr scheduler
                self.lr_schedule.step()

            if self.current_result_better(best_valid_score, val_score):
                log_message('Better valid score found...', self.opt.local_rank)
                best_valid_score = val_score
                best_valid_model_states = {"epoch": epoch, "model": self.model.state_dict(), "optim": self.optimizer.state_dict()}
                self.save_current_results(best_valid_model_states, self.best_valid_model_path)
            if self.current_result_better(best_test_score, test_score):
                log_message('Better test score found...', self.opt.local_rank)
                best_test_score = test_score
                best_test_model_states = {"epoch": epoch, "model": self.model.state_dict(), "optim": self.optimizer.state_dict()}
                self.save_current_results(best_test_model_states, self.best_test_model_path)
            
            torch.distributed.barrier()
            epoch_summary = self.build_message(epoch, train_loss, train_score, val_loss, val_score, test_loss, test_score)
            log_message(epoch_summary, self.opt.local_rank)
            self.log_tf_board(epoch, train_loss, train_score, val_loss, val_score, test_loss, test_score)
            
        # Saving results
        log_message("Training complete.", self.opt.local_rank)
        if self.writer is not None:
            self.writer.close()
        self.log_best_scores(best_valid_score, best_test_score)
        self.save_results(best_valid_model_states, best_test_model_states)

    def prepare_checkpoint_log(self):
        if self.opt.local_rank !=0:
            return None, None, None, None, None
        task_path = os.path.join('./Running/', self.opt.task_name)
        best_valid_model_path = os.path.join(task_path, "best_valid_model.pth.tar")
        best_test_model_path = os.path.join(task_path, "best_test_model.pth.tar")
        newest_model_path = os.path.join(task_path, "newest_model.pth.tar")

        os.makedirs(task_path, exist_ok=True)
        set_logger(os.path.join(task_path, "Running.log"))
        
        writer = SummaryWriter(task_path)
        return task_path, writer, best_valid_model_path, best_test_model_path, newest_model_path
    
    def freeze_params(self):
        for name, param in self.model.named_parameters():
            if 'text_model' in name or 'vision_encoder' in name or 'vision_projection' in name:
                if 'custom' in name or 'embed_tokens' in name or 'lm_head' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True
                            
            if self.opt.print_params:
                msg = '{}\t{}/{}/{}'.format(name, param.requires_grad, str(param.dtype), str(param.shape))
                log_message(msg, self.opt.local_rank)
    
    def get_optimizer(self, model):
        prompt_params, gdecoder_params, ldecoder_params, embedding_params = [], [], [], []
        for p in model.named_parameters():
            if p[1].requires_grad and 'decoder_global' in p[0]:
                gdecoder_params.append(p[1])
            elif p[1].requires_grad and 'decoder_local' in p[0]:
                ldecoder_params.append(p[1])
            elif p[1].requires_grad and ('lm_head' in p[0] or 'embed_tokens' in p[0]):
                embedding_params.append(p[1])
            elif p[1].requires_grad:
                prompt_params.append(p[1])
            else:
                pass
        params = [
            {'params': prompt_params, 'lr': self.opt.prompt_lr},
            {'params': ldecoder_params, 'lr': self.opt.ldecoder_lr},
            {'params': gdecoder_params, 'lr': self.opt.gdecoder_lr},
            {'params': embedding_params, 'lr': self.opt.embed_lr},
        ]
        
        if self.opt.optm == "Adam":
            # optimizer = torch.optim.Adam(params, weight_decay=self.opt.weight_decay,)# eps=1e-4)
            optimizer = bitsandbytes.optim.Adam(params, weight_decay=self.opt.weight_decay, is_paged=True, betas=(0.9, 0.95))# eps=1e-4)
            # optimizer = bitsandbytes.optim.Adam8bit(params, weight_decay=self.opt.weight_decay, is_paged=True, betas=(0.9, 0.95))# eps=1e-4)
        elif self.opt.optm == "SGD":
            optimizer = torch.optim.SGD(params, weight_decay=self.opt.weight_decay, momentum=0.9)
        elif self.opt.optm == "AdamW":
            # optimizer = torch.optim.AdamW(params, weight_decay=self.opt.weight_decay, amsgrad=self.opt.amsgrad, )#eps=1e-4)
            optimizer = bitsandbytes.optim.AdamW(params, weight_decay=self.opt.weight_decay, amsgrad=self.opt.amsgrad, is_paged=True, betas=(0.9, 0.95))#eps=1e-4)
            # optimizer = bitsandbytes.optim.AdamW8bit(params, weight_decay=self.opt.weight_decay, amsgrad=self.opt.amsgrad, is_paged=True, betas=(0.9, 0.95))#eps=1e-4)
        else:
            raise NotImplementedError

        if self.opt.warmup_steps > 0:
            lr_schedule = get_cosine_schedule_with_warmup(optimizer, 
                        num_warmup_steps=self.opt.warmup_steps, num_training_steps=self.opt.epochs_num*len(self.data_loader_train))
        else:
            lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.opt.lr_decrease_iter, self.opt.lr_decrease_rate)
        return optimizer, lr_schedule

    def train(self, train_loader):
        self.model.train()
        running_loss = 0.0
        
        all_ious, all_interactions, all_unions = 0, 0, 0
        precision_iou_threshold, precision_iou_correct, precision_iou = [0.5, 0.6, 0.7, 0.8, 0.9], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
        all_dices = 0
        num_lcl_samples = 0
        
        glb_mae = 0
        num_glb_samples = 0

        for i, datas in enumerate(train_loader):
            images, texts_question, texts_answer, segment_labels, generate_labels, tasks = datas
            texts_input_ids, texts_attention_mask, label_input_ids = get_complete_input_text(texts_question, texts_answer, self.opt.tokenizer_question, self.opt.tokenizer_answer)
            
            images = images.cuda()
            texts_input_ids = texts_input_ids.to(self.opt.device)
            texts_attention_mask = texts_attention_mask.to(self.opt.device)
            segment_labels = segment_labels.to(self.opt.device)
                        
            outputs = self.model(images, texts_input_ids, texts_attention_mask, label_input_ids, generate_labels, segment_labels, tasks, inference=False)
            loss = outputs['final_loss']
            self.optimizer.zero_grad()
            
            # self.scaler.scale(loss).backward()
            loss.backward()
            
            if self.opt.check_gradient:
                self.check_gradient()
            if self.opt.gradient_clip > 0:
                torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.opt.gradient_clip)
            if self.opt.gradient_norm > 0:
                torch.nn.utils.clip_grad_norm_([param for param in self.model.parameters() if param.requires_grad], self.opt.gradient_norm)
            
            # self.scaler.step(self.optimizer)
            # self.scaler.update()
            self.optimizer.step()
            if self.opt.warmup_steps > 0:
                self.lr_schedule.step()
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                running_loss += loss.item()
                
                # Calculate some metrics for segmentation
                lcl_token_generated_results = outputs['lcl_token_generated_results'].detach().cpu()
                
                ious_batch, interactions_batch, unions_batch = compute_IoU_2class_batch(lcl_token_generated_results.cpu(), segment_labels.cpu())
                all_ious, all_interactions, all_unions = all_ious+ious_batch.sum().item(), all_interactions+interactions_batch.sum().item(), all_unions+unions_batch.sum().item()
                all_dices = all_dices + compute_dice_from_iou(ious_batch).sum().item()
                for n_threshold in range(len(precision_iou_threshold)):
                    precision_iou_correct[n_threshold] += (ious_batch >= precision_iou_threshold[n_threshold]).long().sum().item()
                num_lcl_samples += segment_labels.shape[0]
                
                # Calculate some metrics for generation
                glb_mae += outputs['global_generation_mae']
                num_glb_samples += 1
            
            if i % self.opt.print_iter == 0:
                log_message('Iter-{}[{}/{}]: loss[{:5.2f}] OIoU[{:5.2f}] GMAE[{:5.4f}] TLoss[{:5.2f}] LLoss[{:5.2f}] GLoss[{:5.4f}] LR[{:.6f}]'.format(
                    'Train', i, len(train_loader), loss.item(), (all_interactions / all_unions), (glb_mae / num_glb_samples), outputs['text_generation_loss'].cpu().item(), outputs['local_generation_loss'].cpu().item(), outputs['global_generation_loss'].cpu().item(), self.lr_schedule.get_last_lr()[-1]), self.opt.local_rank)
            
            if self.opt.check_gradient:
                exit(0)
        
        with torch.no_grad():
            # torch.tensor is used to cast value to tensors, torch.Tensor recieves shapes and random generate some.
            running_loss = running_loss / len(train_loader)
            running_loss     = torch.tensor(running_loss).to(self.opt.device)
            all_ious         = torch.tensor(all_ious).to(self.opt.device)
            all_interactions = torch.tensor(all_interactions).to(self.opt.device)
            all_unions       = torch.tensor(all_unions).to(self.opt.device)
            all_dices        = torch.tensor(all_dices).to(self.opt.device)
            num_lcl_samples  = torch.tensor(num_lcl_samples).to(self.opt.device)
            precision_iou_correct = torch.tensor(precision_iou_correct).to(self.opt.device)
            num_glb_samples  = torch.tensor(num_glb_samples).to(self.opt.device)
            glb_mae          = torch.tensor(glb_mae).to(self.opt.device)
            self.reduce_metrics([running_loss], [all_ious, all_interactions, all_unions, all_dices, num_lcl_samples, precision_iou_correct, num_glb_samples, glb_mae])
            
            all_ious_mean = (all_ious / num_lcl_samples).item()
            overall_iou = (all_interactions / all_unions).item()
            for n_precision in range(len(precision_iou)):
                precision_iou[n_precision] = precision_iou_correct[n_precision] / num_lcl_samples
            all_dices_mean = (all_dices / num_lcl_samples).item()
            all_glb_mae_mean = (glb_mae / num_glb_samples).item()
                    
        train_score = self.get_score_from_result(all_ious_mean, overall_iou, precision_iou, precision_iou_threshold, all_dices_mean, all_glb_mae_mean, running_loss)

        return running_loss, train_score

    def evaluate(self, valid_loader):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
        
            all_ious, all_interactions, all_unions = 0, 0, 0
            precision_iou_threshold, precision_iou_correct, precision_iou = [0.5, 0.6, 0.7, 0.8, 0.9], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
            all_dices = 0
            num_lcl_samples = 0
            
            glb_mae = 0
            num_glb_samples = 0

            for i, datas in enumerate(valid_loader):
                images, texts_question, texts_answer, segment_labels, generate_labels, tasks = datas
                texts_input_ids, texts_attention_mask, label_input_ids = get_complete_input_text(texts_question, texts_answer, self.opt.tokenizer_question, self.opt.tokenizer_answer)
                
                images = images.cuda()
                texts_input_ids = texts_input_ids.to(self.opt.device)
                texts_attention_mask = texts_attention_mask.to(self.opt.device)
                segment_labels = segment_labels.to(self.opt.device)
                            
                outputs = self.model(images, texts_input_ids, texts_attention_mask, label_input_ids, generate_labels, segment_labels, tasks, inference=False)
                loss = outputs['final_loss']
                
                running_loss += loss.item()
                lcl_token_generated_results = outputs['lcl_token_generated_results'].cpu()
                
                # Calculate some metrics for segmentation
                ious_batch, interactions_batch, unions_batch = compute_IoU_2class_batch(lcl_token_generated_results.cpu(), segment_labels.cpu())
                all_ious, all_interactions, all_unions = all_ious+ious_batch.sum().item(), all_interactions+interactions_batch.sum().item(), all_unions+unions_batch.sum().item()
                all_dices = all_dices + compute_dice_from_iou(ious_batch).sum().item()
                for n_threshold in range(len(precision_iou_threshold)):
                    precision_iou_correct[n_threshold] += (ious_batch >= precision_iou_threshold[n_threshold]).long().sum().item()
                num_lcl_samples += segment_labels.shape[0]
                
                # Calculate some metrics for generation
                glb_mae += outputs['global_generation_mae']
                num_glb_samples += 1
            
            # torch.tensor is used to cast value to tensors, torch.Tensor recieves shapes and random generate some.
            running_loss = running_loss / len(valid_loader)
            running_loss     = torch.tensor(running_loss).to(self.opt.device)
            all_ious         = torch.tensor(all_ious).to(self.opt.device)
            all_interactions = torch.tensor(all_interactions).to(self.opt.device)
            all_unions       = torch.tensor(all_unions).to(self.opt.device)
            all_dices        = torch.tensor(all_dices).to(self.opt.device)
            num_lcl_samples  = torch.tensor(num_lcl_samples).to(self.opt.device)
            precision_iou_correct = torch.tensor(precision_iou_correct).to(self.opt.device)
            num_glb_samples  = torch.tensor(num_glb_samples).to(self.opt.device)
            glb_mae          = torch.tensor(glb_mae).to(self.opt.device)
            self.reduce_metrics([running_loss], [all_ious, all_interactions, all_unions, all_dices, num_lcl_samples, precision_iou_correct, glb_mae, num_glb_samples])
            
            all_ious_mean = (all_ious / num_lcl_samples).item()
            overall_iou = (all_interactions / all_unions).item()
            for n_precision in range(len(precision_iou)):
                precision_iou[n_precision] = precision_iou_correct[n_precision] / num_lcl_samples
            all_dices_mean = (all_dices / num_lcl_samples).item()
            all_glb_mae_mean = (glb_mae / num_glb_samples).item()
            
            valid_score = self.get_score_from_result(all_ious_mean, overall_iou, precision_iou, precision_iou_threshold, all_dices_mean, all_glb_mae_mean, running_loss)

            return running_loss, valid_score

    def reduce_metrics(self, avg_metrics, sum_metrics):
        for metric in avg_metrics:
            dist.all_reduce(metric, dist.ReduceOp.AVG)
        for metric in sum_metrics:
            dist.all_reduce(metric, dist.ReduceOp.SUM)

    def check_gradient(self):
        for name, parms in self.model.named_parameters():
            if parms.requires_grad:
                msg = 'Name/GradRequire/Param/Grad: {}/{}/{:5.2f}/{:5.2f}'.format(name, parms.requires_grad, parms.sum(), parms.grad.sum() if parms.grad is not None else 'NOGRAD')
                log_message(msg, self.opt.local_rank)

    def get_score_from_result(self, all_ious_mean, overall_iou, precision_iou, precision_iou_threshold, dice_mean, mae_mean, loss):
        score = {
            'Loss': loss,
            'MIoU': all_ious_mean,
            'OIoU': overall_iou,
            'MDICE': dice_mean,
            'MGMAE': mae_mean,
        }
        for threshold, precision in zip(precision_iou_threshold, precision_iou):
            score['Pcs@{:.1f}'.format(threshold)] = precision
        return score
        
    def current_result_better(self, best_score, current_score):
        if best_score is None:
            return True
        else:
            return current_score['OIoU'] > best_score['OIoU']

    def build_message(self, epoch, train_loss, train_score, val_loss, val_score, test_loss, test_score):
        msg = "Epoch:[{:3.0f}]\n".format(epoch + 1)       
        
        msg += "Train:"
        msg += " Loss:[{0:.3f}]".format(train_loss)
        msg += " MIoU/OIoU/MDICE/MGMAE:[{0:6.3f}/{1:6.3f}/{2:6.3f}/{3:6.3f}]".format(train_score['MIoU'], train_score['OIoU'], train_score['MDICE'], train_score['MGMAE'])
        msg += " Pcs@0.5/0.6/0.7/0.8/0.9:[{0:6.3f}/{1:6.3f}/{2:6.3f}/{3:6.3f}/{4:6.3f}]".format(train_score['Pcs@0.5'], train_score['Pcs@0.6'], train_score['Pcs@0.7'], train_score['Pcs@0.8'], train_score['Pcs@0.9'])
        msg += '\n'
        
        msg += "Valid:"
        msg += " Loss:[{0:.3f}]".format(val_loss)
        msg += " MIoU/OIoU/MDICE/MGMAE:[{0:6.3f}/{1:6.3f}/{2:6.3f}/{3:6.3f}]".format(val_score['MIoU'], val_score['OIoU'], val_score['MDICE'], val_score['MGMAE'])
        msg += " Pcs@0.5/0.6/0.7/0.8/0.9:[{0:6.3f}/{1:6.3f}/{2:6.3f}/{3:6.3f}/{4:6.3f}]".format(val_score['Pcs@0.5'], val_score['Pcs@0.6'], val_score['Pcs@0.7'], val_score['Pcs@0.8'], val_score['Pcs@0.9'])
        msg += '\n'
        
        msg += "Test:"
        msg += " Loss:[{0:.3f}]".format(test_loss)
        msg += " MIoU/OIoU/MDICE/MGMAE:[{0:6.3f}/{1:6.3f}/{2:6.3f}/{3:6.3f}]".format(test_score['MIoU'], test_score['OIoU'], test_score['MDICE'], test_score['MGMAE'])
        msg += " Pcs@0.5/0.6/0.7/0.8/0.9:[{0:6.3f}/{1:6.3f}/{2:6.3f}/{3:6.3f}/{4:6.3f}]".format(test_score['Pcs@0.5'], test_score['Pcs@0.6'], test_score['Pcs@0.7'], test_score['Pcs@0.8'], test_score['Pcs@0.9'])
        msg += '\n'
            
        return msg.strip()

    def build_single_message(self, best_score, mode):
        msg = mode
        for key in best_score.keys():
            msg += " "+key+":[{0:6.3f}]".format(best_score[key])
        return msg

    def log_tf_board(self, epoch, train_loss, train_score, val_loss, val_score, test_loss, test_score):
        if self.opt.local_rank != 0:
            return
        self.writer.add_scalar('Train/Loss', train_loss, epoch)
        for key in train_score.keys():
            self.writer.add_scalar('Train/'+key, train_score[key], epoch)
            
        self.writer.add_scalar('Valid/Loss', val_loss, epoch)
        for key in val_score.keys():
            self.writer.add_scalar('Valid/'+key, val_score[key], epoch)
        
        self.writer.add_scalar('Test/Loss', test_loss, epoch)
        for key in test_score.keys():
            self.writer.add_scalar('Test/'+key, test_score[key], epoch)
        
        self.writer.add_scalar('Lr',  self.lr_schedule.get_last_lr()[-1], epoch)

    def log_best_scores(self, best_valid_score, best_test_score):
        log_message(self.build_single_message(best_valid_score, 'Best Valid Score: \t'), self.opt.local_rank)
        log_message(self.build_single_message(best_test_score, 'Best Test Score: \t'), self.opt.local_rank)

    def save_results(self, best_valid_model_states, best_test_model_states):
        if self.opt.local_rank != 0:
            return
        torch.save(best_valid_model_states, self.best_valid_model_path)
        torch.save(best_test_model_states, self.best_test_model_path)

    def save_current_results(self, newst_model_states, model_path):
        if self.opt.local_rank != 0:
            return
        torch.save(newst_model_states, model_path)