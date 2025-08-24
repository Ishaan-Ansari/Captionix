import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertTokenizer

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        # resnet = models.resnet50(pretrained=True)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # freeze resnet parameters to prevent updating during training
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-2]  # remove the last two layers (avgpool and fc)
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        features = self.resnet(images)
        # reshape features for attention
        features = features.permute(0, 2, 3, 1)  # (batch_size, height, width, channels)
        features = features.view(features.size(0), -1, features.size(-1))
        return features
    
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.attention_dim = attention_dim
        # linear layers for attention 
        self.W = nn.Linear(encoder_dim, attention_dim)
        self.U = nn.Linear(decoder_dim, attention_dim)
        self.A = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        # calculate attention weights
        u_hs = self.U(hidden_state)
        w_ah = self.W(features)
        combined_states = torch.tanh(u_hs.unsqueeze(1) + w_ah)
        attention_scores = self.A(combined_states)  # (batch_size, seq_len)
        attention_scores = attention_scores.squeeze(2)    
        alpha = F.softmax(attention_scores, dim=1)  # normalize scores
        # apply attention weights to features
        attention_weights = features * alpha.unsqueeze(2)
        attention_weights = attention_weights.sum(dim=1)  
        return alpha, attention_weights

class DecoderRNN(nn.Module):    
    def __init__(self, embed_size, attention_dim, encoder_dim, decoder_dim, vocab_size=None, dropout=0.3):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        # embedding layer using BERT tokenizer vocabulary
        # self.embedding = nn.Embedding(len(BertTokenizer.from_pretrained('bert-base-uncased')), embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size) 
        # initialize hidden state and cell state
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        # LSTM cell for sequence generation
        self.lstm_cell = nn.LSTMCell(embed_size+ encoder_dim, decoder_dim, bias=True)
        # fully connected layer to generate output vocabulary
        self.fcn = nn.Linear(decoder_dim, self.embedding.num_embeddings)
        self.drop = nn.Dropout(dropout)

    def forward(self, features, captions):
        captions_for_embedding = captions[:, :-1]  

        batch_size = features.size(0)
        embeds = self.embedding(captions_for_embedding)  # (batch_size, seq_len, embed_size)
        seq_length = captions_for_embedding.size(1)
        num_features = features.size(1)

        h, c = self.init_hidden_state(features)

        # initialize tensors to store predictions and attention weights
        preds = torch.zeros(batch_size, seq_length, self.fcn.out_features).to(features.device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(features.device)

        # generate sequence
        for s in range(seq_length):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            preds[:, s] = output
            alphas[:, s] = alpha

        return preds, alphas
    
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c
    
class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, dropout=0.3):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(
            embed_size=embed_size, 
            vocab_size=vocab_size,
            attention_dim=attention_dim, 
            encoder_dim=encoder_dim, 
            decoder_dim=decoder_dim, 
            dropout=dropout
        )

    def forward(self, images, captions):
        # encode images and decode captions
        features = self.encoder(images)
        outputs, alphas = self.decoder(features, captions)
        return outputs, alphas
    
if __name__ == "__main__":
    # Example usage
    embed_size = 256
    vocab_size = 10000  # Example vocabulary size
    attention_dim = 128
    encoder_dim = 2048  # ResNet50 output dimension
    decoder_dim = 512

    model = EncoderDecoder(embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, dropout=0.3)
    
    batch_size = 4
    seq_length = 20
    images = torch.randn(batch_size, 3, 224, 224)  
    captions = torch.randint(0, vocab_size, (batch_size, seq_length))  

    outputs, alphas = model(images, captions)
    print(outputs.shape)  
    print(alphas.shape)