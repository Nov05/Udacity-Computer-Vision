import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        # super(EncoderCNN, self).__init__()
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # remove the top fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features) # [batch size, embed size]
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, 
                 p_dropout=0.1):
        super().__init__()

        self.hidden_size = hidden_size

        # The decoder will embed the inputs before feeding them to the LSTM.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            # padding_idx=dictionary.pad(),
            )
        self.dropout = nn.Dropout(p=p_dropout)

        self.lstm = nn.LSTM(
            # For the first layer we'll concatenate the Encoder's final hidden
            # state with the embedded target tokens.
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)
    
        self.linear = nn.Linear(hidden_size, vocab_size)

        # initialize the hidden state (see code below)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        ''' At the start of training, we need to initialize a hidden state;
            there will be none because the hidden state is formed based on perviously seen data.
            So, this function defines a hidden state with all random numbers and of a specified size.'''
        # The axes dimensions are [n_layers, batch_size, hidden_size]
        return (torch.randn(1, 1, self.hidden_size),
                torch.randn(1, 1, self.hidden_size))

    def forward(self, features, captions):
        # shape of features: [batch_size, embed_size], e.g. [10, 256]
        # shape of captions: [batch_size, sequence_size], e.g. [10, 20]
       
        # Embed the target sequence, which has been shifted right by one
        # position and now starts with the end-of-sentence symbol.
        # shape of caption embedded: [10, 20, 256]
        embedded = self.embedding(captions)
        embedded = self.dropout(embedded)
        lstm_input = torch.cat((features.unsqueeze(1), embedded), dim=1)
        lstm_input = lstm_input[:, :-1, :] # remove the last token in the sequence

        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        # LSTM input shape: [batch_size, sequence_size, input_size], e.g. [10, 20, 256]
        # LSTM output shape: [batch_size, sequence_size, hidden_size], e.g. [10, 20, 512]
        lstm_output, _ = self.lstm(lstm_input) # LSTM output, hidden state

        # shape of output: [batch_size, sequence_size, vocab_size], e.g.[10, 20, 9955]
        output = self.linear(lstm_output)
        output = F.log_softmax(output, dim=2)
        
        return output

    def sample(self, features, states=None, max_len=20):
        '''accepts pre-processed image tensor (features) and returns 
           predicted sentence (list of tensor ids of length max_len)'''
        
        # Inference: There are multiple approaches that can be used
        # to generate a sentence given an image, with NIC. The first
        # one is Sampling where we just sample the first word according 
        # to p1, then provide the corresponding embedding
        # as input and sample p2, continuing like this until we sample
        # the special end-of-sentence token or some maximum length.
        # https://arxiv.org/pdf/1411.4555.pdf
        
        # shape of features: torch.Size([1, 256])
        # shape of word_embedding [1, 1, 256]
        lstm_input = features.unsqueeze(1)
        
        idxs = []
        for _ in range(max_len):
            lstm_output, states = self.lstm(lstm_input, states)
            output = self.linear(lstm_output)
            _, idx = torch.max(output[0][0], 0)
            idxs.append(idx.item())
            # embedding input shape [batch_size, sequence_size]
            # embedding output shape [batch_size, sequence_size, embed_size]
            lstm_input = self.embedding(idx.unsqueeze(0).unsqueeze(0))

        return idxs
