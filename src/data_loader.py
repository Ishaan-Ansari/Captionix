import numpy as np
import pandas as pd
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir,  captions_file, tokenizer, transform=None):
        self.root_dir = root_dir
        self.captions_file = captions_file
        self.tokenizer = tokenizer
        self.transform = transform
        
    def __len__(self):
        return len(self.captions_file)
    
    def __getitem__(self, idx):
        img_name = self.captions_file.iloc[idx, 0]
        caption = self.captions_file.iloc[idx, 1]

        img_path = f"{self.root_dir}/{img_name}"
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
        captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from transformers import BertTokenizer

# using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the dataset
root_dir = '/data/images'
captions_file = '/data/captions.txt'
dataset = ImageCaptionDataset(root_dir=root_dir, captions_file=captions_file, tokenizer=tokenizer, transform=transform)

# a subset of the dataset with 5000 samples
subset_indices = list(range(5000)
subset = Subset(dataset, subset_indices)

train_indices, test_indices = train_test_split(subset_indices, train_size=train_size, test_sizze=test_size, random_state=42)

train_subset = Subset(subset, train_indices)
test_subset = Subset(subset, test_indices)

# Create DataLoaders
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=cusstom_collate_fn)
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

