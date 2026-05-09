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