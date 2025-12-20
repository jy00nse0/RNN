import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from constants import LSTM, GRU

# TODO comments, implementation

init_map = {
    'zeros': lambda args: ZerosInit(args.decoder_num_layers, args.decoder_hidden_size, args.encoder_rnn_cell),
    'bahdanau': lambda args: BahdanauInit(args.encoder_hidden_size, args.decoder_num_layers, args.decoder_hidden_size,
                                          args.decoder_rnn_cell),
    'adjust_pad': None,  # TODO
    'adjust_all': None   # TODO
}


def decoder_init_factory(args):
    if args.decoder_init_type == 'bahdanau' and not args.encoder_bidirectional:
        raise AttributeError('Bahdanau decoder init requires encoder to be bidirectional.')
    return init_map[args.decoder_init_type](args)


class DecoderInit(ABC, nn.Module):
    @abstractmethod
    def forward(self, h_n):
        raise NotImplementedError


class ZerosInit(DecoderInit):

    def __init__(self, decoder_num_layers, decoder_hidden_size, rnn_cell_type):
        assert rnn_cell_type == LSTM or rnn_cell_type == GRU
        super(ZerosInit, self).__init__()
        self.decoder_num_layers = decoder_num_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.rnn_cell_type = rnn_cell_type

    def forward(self, h_n):
        # Accept both tensor and LSTM tuple (h_n, c_n)
        if isinstance(h_n, tuple):
            h_n = h_n[0]

        batch_size = h_n.size(1)
        # preserve dtype/device of encoder hidden
        dtype = h_n.dtype
        device = h_n.device
        hidden = torch.zeros(self.decoder_num_layers, batch_size, self.decoder_hidden_size, device=device, dtype=dtype)
        return hidden if self.rnn_cell_type == GRU else (hidden, hidden.clone())


class BahdanauInit(DecoderInit):
    def __init__(self, encoder_hidden_size, decoder_num_layers, decoder_hidden_size, rnn_cell_type):
        super(BahdanauInit, self).__init__()
        assert rnn_cell_type == LSTM or rnn_cell_type == GRU
        # encoder_hidden_size here is expected to be the *per-direction* encoder hidden size
        # (i.e. the original encoder hidden size, not doubled). The code will try to handle both
        # interleaved (num_layers * num_directions, batch, enc_h) and combined
        # (num_layers, batch, enc_h * num_directions) formats.
        self.linear = nn.Linear(in_features=encoder_hidden_size, out_features=decoder_hidden_size)
        self.decoder_num_layers = decoder_num_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.rnn_cell_type = rnn_cell_type

    def forward(self, h_n):
        # Accept both tensor and LSTM tuple (h_n, c_n)
        if isinstance(h_n, tuple):
            h_n = h_n[0]

        num_hidden_states = h_n.size(0)
        batch_size = h_n.size(1)
        hidden_dim = h_n.size(2)

        # Determine format and extract backward states
        # Case A: interleaved PyTorch format: (num_layers * 2, batch, enc_h) where ordering is
        #        [layer0_fwd, layer0_bwd, layer1_fwd, layer1_bwd, ...]
        if num_hidden_states % 2 == 0 and hidden_dim == self.linear.in_features:
            backward_h = h_n[torch.arange(1, num_hidden_states, 2)]  # (num_layers, batch, enc_h)
        # Case B: combined format used elsewhere: (num_layers, batch, enc_h * 2)
        elif num_hidden_states == self.decoder_num_layers and hidden_dim % 2 == 0 and (hidden_dim // 2) == self.linear.in_features:
            # backward part is the second half of last dim
            enc_h = hidden_dim // 2
            backward_h = h_n[:, :, enc_h:]  # (num_layers, batch, enc_h)
        else:
            # Fallback: try best-effort to extract backward states or raise informative error
            try:
                # try interleaved selection as a fallback
                backward_h = h_n[torch.arange(1, num_hidden_states, 2)]
            except Exception:
                raise RuntimeError(
                    "Unable to interpret encoder hidden state shape in BahdanauInit. "
                    f"h_n.shape={tuple(h_n.shape)}, expected either (num_layers*2, batch, enc_h) or "
                    f"(num_layers, batch, enc_h*2)."
                )

        hidden = torch.tanh(self.linear(backward_h))
        hidden = self.adjust_hidden_size(hidden)
        batch_device = h_n.device
        if self.rnn_cell_type == GRU:
            return hidden
        else:
            # For LSTM, create zero-initialized cell state with matching dtype/device
            dtype = h_n.dtype
            cell = torch.zeros(self.decoder_num_layers, batch_size, self.decoder_hidden_size, device=batch_device, dtype=dtype)
            return hidden, cell

    def adjust_hidden_size(self, hidden):
        """
        If encoder and decoder have different number of layers adjust size of initial hidden state for decoder
        by padding with zeros (when decoder has more layers) or slicing hidden state (when encoder has more layers)
        """
        num_layers = hidden.size(0)
        batch_size = hidden.size(1)
        hidden_size = hidden.size(2)

        if num_layers < self.decoder_num_layers:
            hidden = torch.cat([hidden, torch.zeros(self.decoder_num_layers - num_layers, batch_size, hidden_size, device=hidden.device, dtype=hidden.dtype)],
                               dim=0)

        if num_layers > self.decoder_num_layers:
            hidden = hidden[:self.decoder_num_layers]

        return hidden
