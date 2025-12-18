import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from .embeddings import embeddings_factory
from .attention import attention_factory
from .decoder_init import decoder_init_factory

def bahdanau_decoder_factory(args, embed, attn, init, metadata):
    return BahdanauDecoder(
        rnn_cls=getattr(nn, args.encoder_rnn_cell),  # gets LSTM or GRU constructor from nn module
        embed=embed,
        attn=attn,
        init_hidden=init,
        vocab_size=metadata.vocab_size,
        embed_size=args.embedding_size,
        rnn_hidden_size=args.decoder_hidden_size,
        encoder_hidden_size=args.encoder_hidden_size * (2 if args.encoder_bidirectional else 1),
        num_layers=args.decoder_num_layers,
        dropout=args.decoder_rnn_dropout
    )

def luong_decoder_factory(args, embed, attn, init, metadata):
    return LuongDecoder(
        rnn_cls=getattr(nn, args.encoder_rnn_cell),  # gets LSTM or GRU constructor from nn module
        embed=embed,
        attn=attn,
        init_hidden=init,
        vocab_size=metadata.vocab_size,
        embed_size=args.embedding_size,
        rnn_hidden_size=args.decoder_hidden_size,
        attn_hidden_projection_size=args.luong_attn_hidden_size,
        encoder_hidden_size=args.encoder_hidden_size * (2 if args.encoder_bidirectional else 1),
        num_layers=args.decoder_num_layers,
        dropout=args.decoder_rnn_dropout,
        input_feed=args.luong_input_feed
    )

decoder_map = {
    'bahdanau': bahdanau_decoder_factory,
    'luong': luong_decoder_factory
}

def decoder_factory(args, metadata):
    """
    Returns instance of ``Decoder`` based on provided args.
    """
    # TODO: Handle the case where attention type is 'none'
    embed = embeddings_factory(args, metadata)
    attn = attention_factory(args)
    init = decoder_init_factory(args)
    return decoder_map[args.decoder_type](args, embed, attn, init, metadata)


class Decoder(ABC, nn.Module):
    """
    Defines decoder for seq2seq models. Decoder is designed to be iteratively called until caller decides to stop.
    In every step, decoder depends on output in previous timestamps and encoder outputs.

    This class is a base class; concrete implementations must define their own `_forward` methods.

    """

    def __init__(self, *args):
        super(Decoder, self).__init__()
        self._args = []
        self._args_init = {}

    def forward(self, t, input, encoder_outputs, h_n, **kwargs):
        raise NotImplementedError


class LuongDecoder(Decoder):
    """
    Luong decoder for seq2seq models.

    :param rnn_cls: RNN callable constructor. RNN is either LSTM or GRU.
    :param embed: Embedding layer.
    :param attn: Attention layer.
    :param init_hidden: Function for generating initial RNN hidden state.
    :param vocab_size: Size of vocabulary over which we operate.
    :param embed_size: Dimensionality of word embeddings.
    :param rnn_hidden_size: Dimensionality of RNN hidden representation.
    :param attn_hidden_projection_size: Dimensionality of hidden state produced by combining RNN hidden state and
                                attention context. h_att = tanh( W * [c;h_rnn] )
    :param encoder_hidden_size: Dimensionality of encoder hidden representation.
    :param num_layers: Number of layers in RNN. Default: 1.
    :param dropout: RNN dropout layers mask probability. Default: 0.2.
    :param input_feed: If True, input feeding approach will be used.
    Default: False.
    """

    def __init__(self, rnn_cls, embed, attn, init_hidden, vocab_size, embed_size, rnn_hidden_size,
                 attn_hidden_projection_size, encoder_hidden_size, num_layers=1, dropout=0.2, input_feed=False):
        super(LuongDecoder, self).__init__()

        self.input_feed = input_feed
        self.initial_hidden = init_hidden
        self.attn_hidden_projection_size = attn_hidden_projection_size

        self.embed = embed
        
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = rnn_cls(
            input_size=embed_size + (attn_hidden_projection_size if input_feed else 0),
            hidden_size=rnn_hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout
        )
        
        # Define attention mechanism
        self.attn = attn

    def forward(self, t, input, encoder_outputs, h_n, **kwargs):
        # Prepare input word embedding & state
        embedded = self.embed(input)

        # Attention & RNN forward should be optimized now.