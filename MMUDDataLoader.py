import pickle
from torch.utils.data import Dataset
import os


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
        # Returns all fields as a tuple

        return  self.data[idx]

if __name__ == "__main__":
    dataset = MMUDDataset(split='train')
    image_path, caption, enriched_text, sample_texts, sample_masks, sample_boxs, sample_height, sample_width, sample_category, sample_category_text, caption_len, enriched_text_len = dataset[118]
    print(image_path, caption, enriched_text, sample_texts, sample_masks, sample_boxs, sample_height, sample_width, sample_category, sample_category_text, caption_len, enriched_text_len, sep='\n')
