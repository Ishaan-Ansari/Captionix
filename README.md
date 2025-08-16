# Image Captioning 

> [!NOTE]
>  
> ... Idea of building this project stems from curiosity to understand more about CNN, LSTM, and Attention mech.
> 
> ... this is why i decided to build an image-captioning model.
> 
> ... image -> image-cap-model -> caption. 


# Introduction

## Dataset 

**Flickr8k**

- It has around 8k images with their corresponding captions; here I'm using only 5k.

### Data Loader 

Here's what the preprocessing looks like:

```python 
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

        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Tokenize the caption
        caption_tokens = self.tokenizer(caption, padding='max_length', max_length=30, truncation=True, return_tensors='pt')
        caption_tensor = caption_tokens['input_ids'].squeeze(0)  # Remove extra dimension

        return image, caption_tensor
```

## Architecture

> [!IMPORTANT]
>
> Encoder(resnet50) -> extracts features from images
>
> Attention(block) -> Attention Block focuses on important parts of images with respect to captions.
>
> Decoder(lstms) -> generates captions based on image features.


#  

> [!NOTE]


# Reference 

[1] Kelvin Xu, Jimmy Lei Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard S. Zemel, Yoshua Bengio. (April 19 2016). Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.

[2] Andrej Karpathy. (May 21 2105). The Unreasonable Effectiveness of Recurrent Neural Networks.

[3] Lilian Weng. (June 24, 2018). Attention? Attention!

[4] Sagar Vinodababu. A PyTorch Tutorial to Image Captioning.

[5] Image Captioning with Attention by Artyom Makarov.

[6] PyTorch DataLoader: Understand and implement custom collate function by Fabrizio Damicelli

[7] Pytorch Image Captioning Tutorial(wihtout attention) by Aladdin Persson
