import torch
import torch.nn as nn
from util import RNNWrapper
from abc import ABC, abstractmethod
from .embeddings import embeddings_factory


def encoder_factory(args, metadata, embed=None):
    if embed is None:
        embed = embeddings_factory(args, metadata)
    return SimpleEncoder(
        rnn_cls=getattr(nn, args.encoder_rnn_cell),
        embed=embed,
        embed_size=args.embedding_size,
        hidden_size=args.encoder_hidden_size,
        num_layers=args.encoder_num_layers,
        dropout=args.encoder_rnn_dropout,
        bidirectional=args.encoder_bidirectional
    )


class Encoder(ABC, nn.Module):
    """
    Defines encoder for seq2seq model.
    """

    @abstractmethod
    def forward(self, input, h_0=None):
        pass

    @property
    @abstractmethod
    def hidden_size(self):
        pass

    @property
    @abstractmethod
    def bidirectional(self):
        pass

    @property
    @abstractmethod
    def num_layers(self):
        pass


class SimpleEncoder(Encoder):
    """
    Encoder for seq2seq models.
    """

    def __init__(self, rnn_cls, embed, embed_size, hidden_size, num_layers=1, dropout=0.2,
                 bidirectional=False):
        super(SimpleEncoder, self).__init__()

        self._hidden_size = hidden_size
        self._bidirectional = bidirectional
        self._num_layers = num_layers
        self._dropout = dropout

        self.embed = embed

        # [Optimized] cuDNN Fused Implementation
        # PyTorch nn.LSTM's dropout argument implements Zaremba-style dropout
        # (applied to outputs of each layer except the last, NOT to recurrent connections).
        rnn_dropout = dropout if num_layers > 1 else 0.0
        
        self.rnn = rnn_cls(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=rnn_dropout,
            bidirectional=bidirectional
        )

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def bidirectional(self):
        return self._bidirectional

    @property
    def num_layers(self):
        return self._num_layers

    def forward(self, input, h_0=None):
        """
        Inputs:
            input: (seq_len, batch)
            h_0: (num_layers * num_directions, batch, hidden_size) - Optional

        Outputs:
            outputs: (seq_len, batch, hidden_size * num_directions)
            h_n: (num_layers * num_directions, batch, hidden_size)
                 - Returns raw hidden state structure from PyTorch RNN.
                 - No concatenation/projection is done here to maintain flexibility for decoder initialization.
        """
        # [Optimization] Ensure parameter compactness for cuDNN
        # Essential for DataParallel and preventing performance degradation
        self.rnn.flatten_parameters()

        embedded = self.embed(input)
        
        # [Optimized] Call cuDNN fused RNN once
        # outputs: (seq_len, batch, hidden * dir)
        # h_n: (layers * dir, batch, hidden)
        outputs, h_n = self.rnn(embedded, h_0)
        
        return outputs, h_n
