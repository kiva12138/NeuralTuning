import logging
import torch

def str2ints(v): 
    return list(map(int, v.split("-"))) 

def compute_IoU_2class_batch(pred, gt): 
    """The output has 2 dim, and we have two classification tasks.
    pred: [bs, 2, h, w], gt: [bs, h, w]
    """ 
    pred = pred.argmax(1) 
    intersection = torch.sum(torch.mul(pred, gt), dim=(1, 2)) 
    union = torch.sum(torch.add(pred, gt), dim=(1, 2)) - intersection 
    iou = intersection / union 
    iou = torch.nan_to_num(iou, nan=0, posinf=0, neginf=0) 
    return iou, intersection, union 

def compute_dice_from_iou(iou): 
    dice = (2 * iou) / (1 + iou) 
    return dice 

def set_logger(log_path): 
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG) 
    if not logger.handlers: 
        file_handler = logging.FileHandler(log_path) 
        file_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")) 
        logger.addHandler(file_handler) 

        stream_handler = logging.StreamHandler() 
        stream_handler.setFormatter(logging.Formatter("%(message)s")) 
        logger.addHandler(stream_handler) 

def log_message(message, rank): 
    """Logging some message after defining logging and path.""" 
    if rank != 0: 
        return 
    logging.log(msg=message, level=logging.DEBUG) 