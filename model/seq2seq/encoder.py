import torch
import torch.nn as nn
from util import RNNWrapper
from abc import ABC, abstractmethod
from .embeddings import embeddings_factory


def encoder_factory(args, metadata):
    embed = embeddings_factory(args, metadata)
    return SimpleEncoder(
        rnn_cls=getattr(nn, args.encoder_rnn_cell),  # gets LSTM or GRU constructor from nn module
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

    Inputs: input, h_0
        - **input** (seq_length, batch_size): Input sequence.
        - **h_0** (num_layers * num_directions, batch, hidden_size): Initial hidden state of RNN. Default: None.

    Outputs: outputs, h_n
        - **outputs** (seq_len, batch, hidden_size * num_directions): Outputs of RNN last layer for every timestamp.
        - **h_n** (num_layers * num_directions, batch, hidden_size): RNN outputs for all layers for t=seq_len (last
                    timestamp)
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

    :param rnn_cls: RNN callable constructor. RNN is either LSTM or GRU.
    :param embed: Embedding layer.
    :param embed_size: Dimensionality of word embeddings.
    :param hidden_size: Dimensionality of RNN hidden representation.
    :param num_layers: Number of layers in RNN. Default: 1.
    :param dropout: Dropout probability for RNN. Default: 0.2.
    :param bidirectional: If True, RNN will be bidirectional. Default: False.

    Inputs: input, h_0
        - **input** (seq_length, batch_size): Input sequence.
        - **h_0** (num_layers * num_directions, batch, hidden_size): Initial hidden state of RNN. Default: None.

    Outputs: outputs, h_n
        - **outputs** (seq_len, batch, hidden_size * num_directions): Outputs of RNN last layer for every timestamp.
        - **h_n** (num_layers * num_directions, batch, hidden_size): RNN outputs for all layers for t=seq_len (last
                    timestamp)
    """

    def __init__(self, rnn_cls, embed, embed_size, hidden_size, num_layers=1, dropout=0.2,
                 bidirectional=False):
        super(SimpleEncoder, self).__init__()

        self._hidden_size = hidden_size
        self._bidirectional = bidirectional
        self._num_layers = num_layers
        self._dropout = dropout

        self.embed = embed
        # [Paper Reproduction] Zaremba-style dropout:
        # - NO recurrent dropout (dropout=0 in LSTM)
        # - Manual dropout applied between LSTM layers only
        if num_layers > 1:
            # For multi-layer setup, create individual LSTM layers and apply dropout between them
            self.dropout_layer = nn.Dropout(dropout)
            self.rnn_layers = nn.ModuleList()
            for i in range(num_layers):
                layer_input_size = embed_size if i == 0 else hidden_size * (2 if bidirectional else 1)
                self.rnn_layers.append(
                    RNNWrapper(rnn_cls(input_size=layer_input_size,
                                      hidden_size=hidden_size,
                                      num_layers=1,
                                      dropout=0,  # NO dropout inside LSTM
                                      bidirectional=bidirectional))
                )
            self.rnn = None  # Will not use single RNN, use rnn_layers instead
        else:
            # Single layer: no dropout needed
            self.rnn = RNNWrapper(rnn_cls(input_size=embed_size,
                                          hidden_size=hidden_size,
                                          num_layers=num_layers,
                                          dropout=0,  # NO dropout even for single layer
                                          bidirectional=bidirectional))
            self.rnn_layers = None
            self.dropout_layer = None

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
        embedded = self.embed(input)
        
        if self.rnn_layers is not None:
            # Multi-layer with manual dropout between layers
            outputs = embedded
            hidden_states = []
            
            for i, rnn_layer in enumerate(self.rnn_layers):
                # Get initial hidden state for this layer
                layer_h_0 = None
                if h_0 is not None:
                    num_directions = 2 if self._bidirectional else 1
                    layer_h_0 = h_0[i*num_directions:(i+1)*num_directions]
                
                # Apply RNN layer
                outputs, h_n = rnn_layer(outputs, layer_h_0)
                hidden_states.append(h_n)
                
                # Apply dropout between layers (not after the last layer)
                if i < len(self.rnn_layers) - 1:
                    outputs = self.dropout_layer(outputs)
            
            # Concatenate hidden states from all layers
            import torch
            h_n = torch.cat(hidden_states, dim=0)
            return outputs, h_n
        else:
            # Single layer: no dropout
            outputs, h_n = self.rnn(embedded, h_0)
            return outputs, h_n
