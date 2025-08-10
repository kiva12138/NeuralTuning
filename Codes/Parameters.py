import argparse
from Utils import str2ints

def parse_args():
    parser = argparse.ArgumentParser()

    # Names, paths, logs
    parser.add_argument("--task_name", default="test")
    
    # Data parameters
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--persistent_workers", action='store_true')
    parser.add_argument("--pin_memory", action='store_true')
    parser.add_argument("--drop_last", action='store_true')
    parser.add_argument("--image_size", default=224, type=int, choices=[224, 336])
    parser.add_argument("--max_length", default=500, type=int)
    parser.add_argument("--mask_length", default=500, type=int)
    
    # Model parameters
    parser.add_argument("--attention", default='eager', choices=['eager', 'flash_attention_2'])
    parser.add_argument("--float16", action='store_true')
    parser.add_argument("--use_clip_project", action='store_true')
    parser.add_argument("--vision_dtype_half", action='store_true')
    parser.add_argument("--a1", default=1.0, type=float)
    parser.add_argument("--a2", default=1.0, type=float)
    parser.add_argument("--a3", default=1.0, type=float)
    parser.add_argument("--c1", default=0.5, type=float)
    parser.add_argument("--c2", default=0.5, type=float)
    parser.add_argument("--ts_start_layer", default=0, type=int)
    parser.add_argument("--space_project_modules", default='qv', choices=['qv', 'kv', 'qkv'])
    parser.add_argument("--ts_space_dimension", default=512, type=int)
    parser.add_argument("--ts_space_active_rate", default=0.4, type=float)
    parser.add_argument("--ts_space_dropout", default=0.1, type=float)
    parser.add_argument("--ts_ema_alpha", default=0.1, type=float)
    parser.add_argument("--decoder_extract_layers", default='24-18-12', type=str2ints)
    parser.add_argument("--ldecoder_reduce_dim", default=64, type=int)
    parser.add_argument("--ldecoder_num_attention_heads", default=4, type=int)
    parser.add_argument("--ldecoder_dropout", default=0.1, type=float)
    parser.add_argument("--ldecoder_intermediate_size", default=2048, type=int)
    parser.add_argument("--gdecoder_hidden_dim", default=4096, type=int)
    parser.add_argument("--gdecoder_hidden_dropout", default=0.0, type=float)
    parser.add_argument("--print_params", action='store_true')
    parser.add_argument("--print_iter", default=100, type=int)
    parser.add_argument("--checkpoint", action='store_true')
    
    # Training and optimization
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--optm", default="Adam", type=str, choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument("--amsgrad", action='store_true') # For AdamW
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--prompt_lr", default=4e-3, type=float)
    parser.add_argument("--ldecoder_lr", default=4e-3, type=float)
    parser.add_argument("--gdecoder_lr", default=4e-3, type=float)
    parser.add_argument("--embed_lr", default=4e-3, type=float)
    parser.add_argument("--epochs_num", default=2, type=int)
    parser.add_argument("--warmup_steps", default=-1, type=int)
    parser.add_argument("--lr_decrease_iter", default='5-10', type=str2ints)
    parser.add_argument("--lr_decrease_rate", default=0.1, type=float)
    parser.add_argument("--gradient_clip", default=1.0, type=float)
    parser.add_argument("--gradient_norm", default=-1.0, type=float)
    
    parser.add_argument("--check_gradient", action='store_true')

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    args = parse_args()
    print(args)
