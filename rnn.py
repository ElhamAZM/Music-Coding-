import torch
import torch.nn as nn
from constants import *


class Baseline(nn.Module):
  def __init__(self, hidden_size, rnn_cell='lstm', n_layers=1,num_classes = 2 ):
    super(Baseline, self).__init__()
    self.hidden_size = hidden_size
    self.n_layers = n_layers
    self.encoder = nn.Embedding(N_DICT+1, N_DICT+1)  # because of start token, we add +1 on N_DICT

    # TODO: Fill in below
    # Hint: define nn.LSTM / nn.GRU units, with hidden_size and n_layers.
    if rnn_cell == 'lstm':
      self.rnn = nn.LSTM(79, hidden_size, n_layers) #None
    elif rnn_cell == 'gru':
      self.rnn = nn.GRU(79, hidden_size, n_layers) #None

    # TODO: Fill in below
    # input of decoder should be output of rnn,
    # output of decoder should be number of classes
    self.decoder = nn.Linear(in_features= hidden_size, out_features= num_classes)#None  
    self.log_softmax = nn.LogSoftmax(dim=-1)                ##in_feature= size of each input sample here hidden size 
                                                            #out_feature = size of each output sample

  def forward(self, x, hidden, temperature=1.0):
    encoded = self.encoder(x)  # shape of (Batch, N_DICT)
    # To match the RNN input form(step, Batch, Feature), add new axis on first dimension
    encoded = encoded.unsqueeze(0)

    # TODO: Fill in below
    # hint: use self.rnn you made. encoded input and hidden should be fed.
    output, hidden = self.rnn( encoded , hidden)#None 
                                                      
    output = output.squeeze(0)

    # TODO: Fill in below
    # connect output of rnn to decoder.
    # hint: use self.decoder
    output = self.decoder(output)#None 
                          
    # Optional: apply temperature
    # https://cs.stackexchange.com/questions/79241/what-is-temperature-in-lstm-and-neural-networks-generally
    pred = self.log_softmax(output)

    return pred, hidden

  def init_hidden(self, batch_size, random_init=False):
    if random_init:
      return torch.randn(self.n_layers, batch_size, self.hidden_size), \
             torch.randn(self.n_layers, batch_size, self.hidden_size)
    else:
      return torch.zeros(self.n_layers, batch_size, self.hidden_size),\
             torch.zeros(self.n_layers, batch_size, self.hidden_size)
