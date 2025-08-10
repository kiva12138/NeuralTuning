import faulthandler
import random

import os
import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group

from Parameters import parse_args
from Solver import Solver
from Config import CUDA


def set_random_seed(opt):
    # Set random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def ddp_setup(device): 
    init_process_group(backend="nccl")
    torch.cuda.set_device(device)


if __name__ == "__main__":
    faulthandler.enable()

    opt = parse_args()
    set_random_seed(opt)
    
    local_rank = int(os.environ["LOCAL_RANK"])
    gpu_id = CUDA[local_rank]
    device = torch.device('cuda:'+str(gpu_id))
    
    opt.local_rank = local_rank
    opt.gpu_id = gpu_id
    opt.device = device
    opt.world_size = len(CUDA)
    
    ddp_setup(device)
    
    solver = Solver(opt)
    solver.solve()
    
    destroy_process_group()
