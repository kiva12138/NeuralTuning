import pickle
from torch.utils.data import Dataset
import os
from pycocotools import mask
import numpy as np
from PIL import Image

class MMUDDataset(Dataset):
    def __init__(self, split=None, root_dir=r'D:\MMUD'):
        """
        Args:
            split: one of 'train', 'valid', 'test' (loads D:\MMUD\{split}.pkl)
            root_dir: base directory for splits
        """
        assert split in {'train', 'valid', 'test'}, f"Unknown split: {split}"
        split_path = os.path.join(root_dir, f"{split}.pkl")
        self.data = pickle.load(open(split_path, 'rb'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = list(self.data[idx])

        sample_masks = data[4]
        sample_height, sample_width = data[6], data[7]
        sample_masks_new = []
        for mask in sample_masks:
            mask = self.get_mask(mask, sample_height, sample_width)
            sample_masks_new.append(mask)
        
        data[4] = sample_masks_new

        return  tuple(data)
    
    def get_mask(self, sample_mask, height, width):
        if type(sample_mask[0]) == float:
            refer_mask = mask.frPyObjects([sample_mask], height, width) # From COCO dataset
        else:
            refer_mask = sample_mask # From Clef dataset

        refer_mask = mask.decode(refer_mask)
        refer_mask = np.sum(refer_mask, axis=2)
        refer_mask = refer_mask.astype(np.uint8)

        refer_mask_final = np.zeros(refer_mask.shape)
        refer_mask_final[refer_mask == 1] = 1
        refer_mask = Image.fromarray(refer_mask.astype(np.uint8), mode="P")

        return refer_mask


if __name__ == "__main__":
    dataset = MMUDDataset(split='train')
    image_path, caption, enriched_text, sample_texts, sample_masks, sample_boxs, sample_height, sample_width, sample_category, sample_category_text, caption_len, enriched_text_len = dataset[22]
    print(image_path, caption, enriched_text, sample_texts, sample_masks, sample_boxs, sample_height, sample_width, sample_category, sample_category_text, caption_len, enriched_text_len, sep='\n')
