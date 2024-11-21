import torch.nn as nn
import torch


class FeedForwardNN(nn.Module):
    """ see torch.nn.Module for documentation.
        here we have inherited the base class and implemented a Feed-Forward NN with linear input and output layer, and a GELU activation function.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(input_size, hidden_size)])
        for i in range(num_layers-1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.gelu = nn.GELU()

    def forward(self, out):
        # passing data forward in the NN
        for layer in self.hidden_layers:
            out = self.gelu(layer(out))
        out = self.output_layer(out)
        return out


class RNN_many_to_one(nn.Module):
    """ 
    (batch_size, seq_length, n_features) as input size
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.input_size = input_size
        self.RNN = nn.RNN(input_size, hidden_size,
                          num_layers, batch_first=True)
        self.dense = nn.linear(hidden_size, num_classes)

    def forward(self, out):
        h0 = torch.zeros(self.num_layers, out.size(0), self.hidden_size)
        out, _ = self.rnn(out, h0)
        out = out[:, -1, :]
        out = self.dense(out)
        return out
