import os
import re
import random
import torch
import pickle
import numpy as np
from PIL import Image
from pycocotools import mask

from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from transformers import CLIPProcessor
from Config import DatasetPath, ClipPath, ClipImageSize, CaptionPrompts, GenerationPrompts, SegmentPrompts, SegmentAnswerPrompts, GenerationAnswerPrompts, ClipGenFeaturePath, ImagePath, ModelQuestionTemplate

import logging
logging.getLogger("PIL").setLevel(logging.WARNING)

# MUST use the two parameters: use_square_size=True, do_center_crop=False
# However, there is a bug in transformers==4.37, so we have to set size=(224, 224) when calling
processor = CLIPProcessor.from_pretrained(ClipPath, use_square_size=True, do_center_crop=False)


class HaruDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train', max_length=500, mask_length=10):
        self.mode = mode
        self.image_size = ClipImageSize[ClipPath]
        self.max_length = max_length
        self.mask_length = mask_length
                
        with open(os.path.join(DatasetPath, mode+'_withlength.pkl'), 'rb') as f:
            all_samples = pickle.load(f)
                        
        # print(len(all_samples))
        new_all_samples = []
        for image_path, caption, enriched_text, sample_texts, sample_masks, sample_boxs, sample_height, sample_width, sample_category, sample_category_text, caption_len, enriched_text_len in all_samples:
            
            if type(sample_masks[0][0]) != float:
                continue
            
            bad_text = False
            for sample_text_list in sample_texts:
                sample_text_list_str = ' '.join(sample_text_list)
                bad_symbols = [',', '.', ':', '?', '!', '"', "'", '#', '@', "$", "%", '/', "\\", "&", "^", "*"]
                for symbol in bad_symbols:
                    if symbol in sample_text_list_str:
                        bad_text = True
                        break
                if bad_text:
                    break
            if bad_text:
                continue
            
            if caption_len > self.max_length:
                continue
            elif enriched_text_len > self.max_length:
                continue
            elif len(sample_masks) > self.mask_length:
                continue
            else:
                new_all_samples.append((image_path, caption, enriched_text, sample_texts, sample_masks, sample_boxs, sample_height, sample_width, sample_category, sample_category_text))
        self.all_samples = new_all_samples
        
        self.transforms_for_mask = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.all_samples)
    
    def get_mask(self, sample_mask, height, width):
        if type(sample_mask[0]) == float:
            refer_mask = mask.frPyObjects([sample_mask], height, width) # COCO dataset
        else:
            refer_mask = sample_mask # Clef dataset

        refer_mask = mask.decode(refer_mask)
        refer_mask = np.sum(refer_mask, axis=2)
        refer_mask = refer_mask.astype(np.uint8)
        return refer_mask

    def __getitem__(self, index):
        sample = self.all_samples[index]
        image_path, caption, enriched_text, sample_texts, sample_masks, sample_boxs, sample_height, sample_width, sample_category, sample_category_text = sample
        
        image_path = image_path.split('\\')[-1]
        image = Image.open(os.path.join(ImagePath, image_path)).convert("RGB")
        image_pixel_values = processor(images=image, return_tensors="pt", size=(self.image_size, self.image_size))['pixel_values'].squeeze()
        
        with open(os.path.join(ClipGenFeaturePath, image_path+'.pkl'), 'rb') as f:
            generation_feature_target = pickle.load(f).squeeze()
                
        refer_masks = []
        for sample_mask in sample_masks:
            refer_mask_ = np.array(self.get_mask(sample_mask, sample_height, sample_width))
            refer_mask = np.zeros(refer_mask_.shape)
            refer_mask[refer_mask_ == 1] = 1
            refer_mask = Image.fromarray(refer_mask.astype(np.uint8), mode="P")
            refer_mask = self.transforms_for_mask(refer_mask)
            refer_masks.append(refer_mask)
        
        return image_pixel_values, caption, enriched_text, sample_texts, refer_masks, generation_feature_target


# This is the main function we build the dataloader for different tasks.
def multi_collate_haru(batch):
    # batch is a list with length: batch_size, each element is a list containing the data returned in dataset.
    # image_pixel_values, caption, enriched_text, sample_texts, refer_masks, generation_feature_target
        
    # Random split tasks
    batch_size = len(batch)
    if batch_size < 8:
        print('Batch size should be larger than 6, as we have three tasks.')
        raise NotImplementedError
    caption_task_mount, generation_task_mount, direct_segment_mount = int(batch_size*0.25), int(batch_size*0.25), int(batch_size*0.25)
    complex_segment_mount = batch_size - caption_task_mount - generation_task_mount - direct_segment_mount
    
    sample_indices = list(range(0, batch_size))
    random.shuffle(sample_indices)
    caption_tasks_indices    = sample_indices[0 : caption_task_mount]
    generation_tasks_indices = sample_indices[caption_task_mount : caption_task_mount+generation_task_mount]
    direct_segment_indices   = sample_indices[caption_task_mount+generation_task_mount : caption_task_mount+generation_task_mount+direct_segment_mount]
    complex_segment_indices  = sample_indices[caption_task_mount+generation_task_mount+direct_segment_mount : ]
    
    new_batch_images = []
    new_batch_texts_question = []
    new_batch_texts_answer = []
    new_batch_segment_labels = []
    new_batch_generation_labels = []
    tasks = [] # Flags for different tasks, 0-caption, 1-generation, 2-segment
    
    for i in range(batch_size):
        image_pixel_values, caption, enriched_text, sample_texts, refer_masks, generation_feature_target = batch[i]
        new_batch_images.append(image_pixel_values)
        
        if i in caption_tasks_indices:
            tasks.append(0)
            
            question = ModelQuestionTemplate.format(random.choice(CaptionPrompts))
            new_batch_texts_question.append(question)
            new_batch_texts_answer.append(caption)
        elif i in generation_tasks_indices:
            tasks.append(1)
            
            question = ModelQuestionTemplate.format(random.choice(GenerationPrompts).format(caption))
            new_batch_texts_question.append(question)
            new_batch_texts_answer.append(random.choice(GenerationAnswerPrompts))
            new_batch_generation_labels.append(generation_feature_target)
        elif i in complex_segment_indices:
            tasks.append(2)
            
            matches = re.findall(r"<LCL-(10|\d{1,2})>", enriched_text)
            matches_unique = sorted(set(matches), key=matches.index)
            
            enriched_text_new = enriched_text
            enriched_text_new = re.sub("##Question:", '', enriched_text_new)
            enriched_text_new = re.sub("\n##Answer:", '', enriched_text_new)
            for j, lcl_match in enumerate(matches_unique):
                enriched_text_new = re.sub("<LCL-{}>".format(lcl_match), '', enriched_text_new)
            for j, lcl_match in enumerate(matches_unique):
                enriched_text_new = enriched_text_new + " <LCL-{}>".format(j+1)
            
            enriched_text_new_split = enriched_text_new.split('<SEP>')
            
            question = ModelQuestionTemplate.format(enriched_text_new_split[0])
            new_batch_texts_question.append(question.strip())
            new_batch_texts_answer.append(enriched_text_new_split[1].strip())
            
            for lcl_match in matches:
                lcl_matched_mask = refer_masks[int(lcl_match)-1]
                lcl_matched_mask = torch.as_tensor(np.asarray(lcl_matched_mask).copy(), dtype=torch.int64)
                new_batch_segment_labels.append(lcl_matched_mask) # matchs starts from 1
        elif i in direct_segment_indices:
            tasks.append(2)
            
            object_texts, object_results = [], []
            for j, (sample_text_list, refer_mask) in enumerate(zip(sample_texts, refer_masks)):
                sample_text = random.choice(sample_text_list)
                object_texts.append(sample_text)
                
                lcl_matched_mask = torch.as_tensor(np.asarray(refer_mask).copy(), dtype=torch.int64)
                new_batch_segment_labels.append(lcl_matched_mask)
                
                object_results.append("<LCL-{}>".format(j+1))
                
            object_texts_str = ', '.join(object_texts)
            object_results_str = ', '.join(object_results)
            question = ModelQuestionTemplate.format(random.choice(SegmentPrompts).format(object_texts_str))
            new_batch_texts_question.append(question)
            
            new_batch_texts_answer.append(random.choice(SegmentAnswerPrompts).format(object_results_str))
        else:
            raise NotImplementedError    
    
    new_batch_images = torch.stack(new_batch_images, dim=0)    
    new_batch_texts_question = new_batch_texts_question
    new_batch_texts_answer = new_batch_texts_answer
    new_batch_segment_labels = torch.stack(new_batch_segment_labels, dim=0)
    new_batch_generation_labels = torch.stack(new_batch_generation_labels, dim=0)
    tasks = tasks
        
    return new_batch_images, new_batch_texts_question, new_batch_texts_answer, new_batch_segment_labels, new_batch_generation_labels, tasks


def get_data_loader_ddp(batch_size, drop_last, num_workers, persistent_workers, pin_memory, max_length=500, mask_length=10):
    dataset_train = HaruDataset(mode='train', max_length=max_length, mask_length=mask_length)
    dataset_val = HaruDataset(mode='valid', max_length=max_length, mask_length=mask_length)
    dataset_test = HaruDataset(mode='test', max_length=max_length, mask_length=mask_length)
    sampler_train = DistributedSampler(dataset_train, shuffle=True,  drop_last=drop_last)
    sampler_val   = DistributedSampler(dataset_val,   shuffle=False, drop_last=drop_last)
    sampler_test  = DistributedSampler(dataset_test, shuffle=False, drop_last=drop_last)
    
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, collate_fn=multi_collate_haru, drop_last=drop_last, sampler=sampler_train,
                                                    num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory, shuffle=False)
    data_loader_val =   torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, collate_fn=multi_collate_haru, drop_last=drop_last, sampler=sampler_val,
                                                    num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory, shuffle=False)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, collate_fn=multi_collate_haru, drop_last=drop_last, sampler=sampler_test,
                                                    num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory, shuffle=False)

    return data_loader_train, data_loader_val, data_loader_test


def get_data_loader(batch_size, drop_last=True, num_workers=4, persistent_workers=False, pin_memory=False, max_length=500, mask_length=10):
    dataset_train = HaruDataset(mode='train', max_length=max_length, mask_length=mask_length)
    dataset_val = HaruDataset(mode='valid', max_length=max_length, mask_length=mask_length)
    dataset_test = HaruDataset(mode='test', max_length=max_length, mask_length=mask_length)
    
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, collate_fn=multi_collate_haru, drop_last=drop_last,
                                                    num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory, shuffle=False)
    data_loader_val =   torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, collate_fn=multi_collate_haru, drop_last=drop_last,
                                                    num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory, shuffle=False)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, collate_fn=multi_collate_haru, drop_last=drop_last,
                                                    num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory, shuffle=False)

    return data_loader_train, data_loader_val, data_loader_test


def get_unnormalized_image(image):
    image = image * torch.tensor(processor.image_processor.image_std)[:, None, None] + torch.tensor(processor.image_processor.image_mean)[:, None, None]
    return image


def get_complete_input_text(texts_question, texts_answer, tokenizer_question, tokenizer_answer):
    tokens_question = tokenizer_question(texts_question, padding=False)
    tokens_answer = tokenizer_answer(texts_answer, padding=False)

    fused_tokens = {
        'input_ids': [],
        'attention_mask': []
    }
    question_end_indices = []
    for token1, token2, mask1, mask2 in zip(tokens_question['input_ids'], tokens_answer['input_ids'], tokens_question['attention_mask'], tokens_answer['attention_mask']):
        fused_tokens['input_ids'].append(token1+token2)
        fused_tokens['attention_mask'].append(mask1+mask2)
        question_end_indices.append(len(token1))

    fused_tokens = tokenizer_question.pad(fused_tokens, return_tensors='pt')
    
    # Fill the padding positions with -100
    label_input_ids = fused_tokens['input_ids'].clone()
    label_input_ids = label_input_ids.masked_fill(fused_tokens['attention_mask']==0, -100)
    
    # Fill the question parts with -100
    batch_size, sequence_len = label_input_ids.shape
    question_end_indices = torch.tensor(question_end_indices)
    question_answer_mask = torch.arange(0, sequence_len, dtype=torch.long).repeat(batch_size, 1) < question_end_indices.unsqueeze(-1)

    label_input_ids = torch.masked_fill(label_input_ids, question_answer_mask, -100).long()
    
    return fused_tokens['input_ids'], fused_tokens['attention_mask'], label_input_ids

