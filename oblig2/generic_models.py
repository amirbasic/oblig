import torch
import torch.nn as nn
import torch.nn.functional as F

cpu = torch.device('cpu')


class AVGPooling(nn.Module):
    "Average pooling used in model"

    def __init__(self):
        super().__init__()

    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.tensor
        mask: torch.tensor

        Returns
        -------
        x: torch.tensor - the reduced tensor after average pooling
        """

        x = torch.sum(x, dim=1) / (mask == False).sum(dim=1).unsqueeze(-1)

        return x


class MAXPooling(nn.Module):
    "Max pooling used in model"

    def __init__(self):
        super().__init__()

    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.tensor
        mask: torch.tensor

        Returns
        -------
        x: torch.tensor - the reduced tensor after max pooling
        """

        x = torch.max(x, dim=1).values

        return x


class SUMPooling(nn.Module):
    "Sum pooling used in model"

    def __init__(self):
        super().__init__()

    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.tensor
        mask: torch.tensor

        Returns
        -------
        x: torch.tensor - the reduced tensor after sum pooling
        """

        x = torch.sum(x, dim=1)

        return x

# BiLSTM forward må gjøres om til å ta imot to argumenter


class BiLSTMLayer(nn.Module):
    """ see torch.nn.Module for documentation.
        here we have inherited the base class and implemented a Bi-LSTM layer.
    """

    def __init__(self, args):
        super().__init__()
        self.bilstm = nn.LSTM(input_size=args.hidden_size,
                              hidden_size=args.hidden_size, bidirectional=True, batch_first=True, dropout=args.dropout)
        self.c_0h = nn.Parameter(torch.zeros(2, 1, args.hidden_size)).float()
        self.c_0p = nn.Parameter(torch.zeros(2, 1, args.hidden_size)).float()
        self.h_0h = nn.Parameter(torch.zeros(2, 1, args.hidden_size)).float()
        self.h_0p = nn.Parameter(torch.zeros(2, 1, args.hidden_size)).float()

    def forward(self, out_p, out_h, length_h, length_p):
        c_0h = self.c_0h.expand(-1, out_h.size(0), -1).contiguous()
        c_0p = self.c_0p.expand(-1, out_p.size(0), -1).contiguous()
        h_0h = self.h_0h.expand(-1, out_h.size(0), -1).contiguous()
        h_0p = self.h_0p.expand(-1, out_p.size(0), -1).contiguous()

        out_h = nn.utils.rnn.pack_padded_sequence(
            out_h, length_h.to(cpu), batch_first=True, enforce_sorted=False)  # pack for more optimized pass
        out_h = self.bilstm(out_h, (h_0h, c_0h))[0]  # put to LSTM
        out_h, _ = nn.utils.rnn.pad_packed_sequence(
            out_h, batch_first=True, padding_value=0.0)  # unpack

        out_p = nn.utils.rnn.pack_padded_sequence(
            out_p, length_p.to(cpu), batch_first=True, enforce_sorted=False)  # pack for more optimized pass
        out_p = self.bilstm(out_p, (h_0p, c_0p))[0]  # put to LSTM
        out_p, _ = nn.utils.rnn.pad_packed_sequence(
            out_p, batch_first=True, padding_value=0.0)  # unpack

        # sum the vectors from both directions
        out_h = out_h.unflatten(-1, (2, -1)).sum(-2)
        out_p = out_p.unflatten(-1, (2, -1)).sum(-2)

        return out_p, out_h


class GRULayer(nn.Module):
    """ see torch.nn.Module for documentation.
        here we have inherited the base class and implemented a GRU layer.
    """

    def __init__(self, args):
        super().__init__()
        self.gru = nn.GRU(input_size=args.hidden_size,
                          hidden_size=args.hidden_size, batch_first=True, dropout=args.dropout)
        self.h_0h = nn.Parameter(torch.zeros(1, 1, args.hidden_size)).float()
        self.h_0p = nn.Parameter(torch.zeros(1, 1, args.hidden_size)).float()

    def forward(self, out_p, out_h):  # , length_h, length_p
        h_0h = self.h_0h.expand(-1, out_h.size(0), -1).contiguous()
        h_0p = self.h_0p.expand(-1, out_p.size(0), -1).contiguous()

        out_p = self.gru(out_p, (h_0p))[0]
        out_h = self.gru(out_h, (h_0h))[0]

        return out_p, out_h


class ElmanLayer(nn.Module):
    """ see torch.nn.Module for documentation.
        here we have inherited the base class and implemented a Elman layer.
    """

    def __init__(self, args):
        super().__init__()
        self.elman = nn.RNN(input_size=args.hidden_size,
                            hidden_size=args.hidden_size, batch_first=True, dropout=args.dropout)
        self.h_0h = nn.Parameter(torch.zeros(1, 1, args.hidden_size)).float()
        self.h_0p = nn.Parameter(torch.zeros(1, 1, args.hidden_size)).float()

    def forward(self, out_p, out_h):
        h_0h = self.h_0h.expand(-1, out_h.size(0), -1).contiguous()
        h_0p = self.h_0p.expand(-1, out_p.size(0), -1).contiguous()

        out_p = self.elman(out_p, (h_0p))[0]
        out_h = self.elman(out_h, (h_0h))[0]

        return out_p, out_h


class FFLayer(nn.Module):
    """ see torch.nn.Module for documentation.
        here we have inherited the base class and implemented a feed-forward layer with Gelu activation function.
    """

    def __init__(self, args):
        super().__init__()
        self.linear = nn.Linear(args.hidden_size, args.hidden_size)
        self.gelu = nn.GELU()

    def forward(self, out_p, out_h):
        out_p = self.gelu(self.linear(out_p))
        out_h = self.gelu(self.linear(out_h))

        return out_p, out_h


class Model(nn.Module):
    """ see torch.nn.Module for documentation.
        here we have inherited the base class and created our base architecture.
        It takes in pooling type, layer type, freeze, dropout hidden size and num of layers as input from the main script.
    """

    def __init__(self, args, word2vec, n_labels):
        super().__init__()
        self.pad_index = word2vec.get_index("[PAD]")
        self.pooling_type = args.pooling_type
        self.layer_type = args.layer_type
        self.pool = {'avg': AVGPooling, 'max': MAXPooling,
                     'sum': SUMPooling}[self.pooling_type]
        self.layer = {'ff': FFLayer, 'elman': ElmanLayer,
                      'gru': GRULayer, 'bilstm': BiLSTMLayer}[self.layer_type]

        self.embedding = nn.Sequential(
            nn.Embedding.from_pretrained(torch.FloatTensor(
                word2vec.vectors), freeze=args.freeze),
            nn.Dropout(args.dropout),
            nn.Linear(word2vec.vector_size, args.hidden_size),
        )

        self.layers = nn.ModuleList([
            self.layer(args)
            for _ in range(args.num_layers)
        ])

        self.pooling = self.pool()
        self.output_layer = nn.Linear(args.hidden_size * 4, n_labels)
        
        self.gelu = nn.GELU()

    def forward(self, premise, hypothesis):
        mask_p = (premise == self.pad_index)
        mask_h = (hypothesis == self.pad_index)

        length_h = (~mask_h).long().sum(dim=1)  # shape: [B]
        length_p = (~mask_p).long().sum(dim=1)  # shape: [B]

        out_p = self.gelu(self.embedding(premise))
        out_h = self.gelu(self.embedding(hypothesis))

        for layer in self.layers:
            if isinstance(layer, BiLSTMLayer):
                out_p, out_h = layer(out_p, out_h, length_h, length_p)
            else:
                out_p, out_h = layer(out_p, out_h)

        out_p = self.pooling(out_p, mask_p)
        out_h = self.pooling(out_h, mask_h)

        combined = torch.cat(
            (out_p, out_h, torch.abs(out_p - out_h), out_p * out_h), 1)

        return self.output_layer(combined)
