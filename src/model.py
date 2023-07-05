import torch
import torchvision

import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    def __init__(self, embed_size) -> None:
        super(CNNEncoder, self).__init__()
        self.inception = models.inception_v3(aux_logits=True)
        self.inception.fc = nn.Linear(
            self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        features, _ = self.inception(input)
        features = self.relu(features)
        return self.dropout(features)


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size) -> None:
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = self.dropout(embeddings)
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class Encoder_Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size) -> None:
        super(Encoder_Decoder, self).__init__()
        self.cnn = CNNEncoder(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.cnn(images)
        outputs = self.decoderRNN(features, captions)
        return outputs
