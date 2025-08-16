import numpy as np
import pandas as pd
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer

class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir,  captions_file, tokenizer, transform=None):
        self.root_dir = root_dir
        if isinstance(captions_file, str):
            if captions_file.endswith('.csv'):
                self.captions_df = pd.read_csv(captions_file)
            else:
                self.captions_df = pd.read_csv(captions_file, sep='\t', header=None, names=['image', 'caption'])
        else:
            self.captions_df = captions_file
        
        self.tokenizer = tokenizer
        self.transform = transform
        
    def __len__(self):
        return len(self.captions_df)
    
    def __getitem__(self, idx):
        img_name = self.captions_df.iloc[idx, 0]
        caption = self.captions_df.iloc[idx, 1]

        # img_path = f"{self.root_dir}/{img_name}"
        img_path = os.path.join(self.root_dir, img_name)


        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Tokenize the caption
        caption_tokens = self.tokenizer(caption, padding='max_length', max_length=30, truncation=True, return_tensors='pt')
        caption_tensor = caption_tokens['input_ids'].squeeze(0)  # Remove extra dimension

        return image, caption_tensor

    def custom_collate_fn(batch):
        images, captions = zip(*batch)
        images = torch.stack(images, dim=0)
        # captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
        
        # Since we're using padding='max_length', all captions should be the same length
        # But we can still use pad_sequence for robustness
        captions = torch.stack(captions, dim=0)  # Stack instead of pad_sequence since they're same length
        return images, captions

# using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load the dataset
root_dir = '/data/images'
captions_file = '/data/captions.txt'
dataset = ImageCaptionDataset(root_dir=root_dir, captions_file=captions_file, tokenizer=tokenizer, transform=transform)

# a subset of the dataset with 5000 samples
subset_indices = list(range(min(5000, len(dataset))))  # Ensure we don't exceed dataset size
subset = Subset(dataset, subset_indices)

# Define train/test split ratios
train_size = 0.8
test_size = 0.2

train_indices, test_indices = train_test_split(subset_indices, train_size=train_size, test_size=test_size, random_state=42)

train_subset = Subset(subset, train_indices)
test_subset = Subset(subset, test_indices)

# Create DataLoaders
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

