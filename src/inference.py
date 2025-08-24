import torch
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from src.model import EncoderDecoder

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# model
embed_size = 300
attention_dim = 256
encoder_dim = 2048
decoder_dim = 512
decoder_dim = 512
drop_prob = 0.3

model = EncoderDecoder(
    embed_size=embed_size,
    attention_dim=attention_dim,
    encoder_dim=encoder_dim,
    decoder_dim=decoder_dim,
    dropout=drop_prob
).to(device)

# load the trained model weights
model.load_state_dict(torch.load('/model_weights.pth', map_location=device))
model.eval()

# image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_caption(image_path, model, tokenizer, transform, max_length=50):
    # load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # encode the image
    features = model.encoder(iamge_tensor)

    # encode the image
    features = model.encoder(image_tensor)

    # initialize the hidden and cell states
    h, c = model.decoder.init_hidden_state(features)

    # start the caption with the [CLS] token
    word = torch.tensor([tokenizer.cls_token_id]).to(device)
    embeds =  model.decoder.embedding(word)

    caption = []
    alphas = []

    for _ in range(max_len):
        alpha, context = model.decoder.attention(features, h)
        alphas.append(alpha.cpu().detach().numpy())

        lstm_input = torch.cat((embeds, context), dim=1)
        h, c = model.decoder.lstm_cell(lstm_input, (h, c))
        

        output = model.decoder.fcn(model.decoder.drop(h))
        predicted_word_idx = output.argmax(dim=1)

        captions.append(predicted_word_idx.item())


        # break if [SEP] token is generated
        if predicted_word_idx.item() == tokenizer.sep_token_id:
            break


        embeds = model.decoder.embedding(predicted_word_idx.unsqueeze(0))

    caption = tokenizer.decode(caption, skip_special_tokens=True)
    return image, caption