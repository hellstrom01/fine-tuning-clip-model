"""
This is handling everything about tokenization and 
the images. 

main function 
"""
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class RSIDCDataset(Dataset):
    def __init__(self, pairs, preprocess, tokenizer):
        self.pairs = pairs # (image,caption) 
        self.preprocess = preprocess # given from the init of the model 
        self.tokens = tokenizer([caption for _, caption in pairs]) # this is the text in the images

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        image_path, _ = self.pairs[idx]
        image = self.preprocess(Image.open(image_path))
        tokens = self.tokens[idx]
        return image, tokens
    

import os

class RSICDLabeledDataset(Dataset):
    """
    Takes a list of image dicts, flattens the 5 captions, 
    and returns (image, text_tokens)
    """
    def __init__(self, image_list, img_dir, transform, tokenizer):
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        
        # Flatten the captions
        self.samples = []
        for item in image_list:
            for sentence in item['sentences']:
                self.samples.append({
                    'filename': item['filename'],
                    'caption': sentence['raw']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.img_dir, sample['filename'])
        
        image = Image.open(img_path)
        image = self.transform(image)
        text_tokens = self.tokenizer([sample['caption']])[0]
        
        return image, text_tokens


class RSICDUnlabeledDataset(Dataset):
    """
    Takes a list of image dicts, applies aggressive SimCLR transform twice,
    and returns (view_1, view_2)
    """
    def __init__(self, image_list, img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = image_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename = self.samples[idx]['filename']
        img_path = os.path.join(self.img_dir, filename)
        
        image = Image.open(img_path)
        
        # Create the two views for Contrastive Learning
        view_1 = self.transform(image)
        view_2 = self.transform(image)
        
        return view_1, view_2
    