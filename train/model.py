import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class EncoderRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_size: int,
        hidden_size: int,
        num_layers: int = 4,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        outputs, (h_n, c_n) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs,
            batch_first=True,
        )
        return outputs, (h_n, c_n)


class GlobalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def score(
        self,
        ht: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.bmm(
            encoder_outputs,
            ht.unsqueeze(2)
        ).squeeze(2)

    def forward(
        self,
        ht: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(ht, encoder_outputs)
        scores = scores.masked_fill(src_mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(
            attn_weights.unsqueeze(1),
            encoder_outputs,
        ).squeeze(1)
        return context, attn_weights


class DecoderWithAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_size: int,
        hidden_size: int,
        num_layers: int = 4,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.input_size = emb_size + hidden_size
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = GlobalAttention(hidden_size)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(
        self,
        y_prev: torch.Tensor,
        h_c: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
        h_tilde_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor],
               torch.Tensor, torch.Tensor]:
        emb = self.embedding(y_prev)
        if h_tilde_prev is None:
            h_tilde_prev = emb.new_zeros(emb.size(0), self.hidden_size)
        lstm_input = torch.cat([emb, h_tilde_prev], dim=-1).unsqueeze(1)
        lstm_output, h_c = self.lstm(lstm_input, h_c)
        h_t = lstm_output.squeeze(1)
        context, attn_weights = self.attention(h_t, encoder_outputs, src_mask)
        attn_input = torch.cat([context, h_t], dim=-1)
        h_tilde = torch.tanh(self.attn_combine(attn_input))
        h_tilde = self.dropout(h_tilde)
        logits = self.out(h_tilde)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, h_c, h_tilde, attn_weights

    def forward(
        self,
        tgt: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        teacher_forcing: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, tgt_len = tgt.size()
        outputs = []
        attn_weights_all = []
        y_prev = tgt[:, 0]
        h_c = hidden
        h_tilde_prev = None
        for t in range(1, tgt_len):
            log_probs, h_c, h_tilde_prev, attn_weights = self.forward_step(
                y_prev, h_c, encoder_outputs, src_mask, h_tilde_prev
            )
            outputs.append(log_probs.unsqueeze(1))
            attn_weights_all.append(attn_weights.unsqueeze(1))
            if teacher_forcing:
                y_prev = tgt[:, t]
            else:
                y_prev = log_probs.argmax(dim=-1)
        outputs = torch.cat(outputs, dim=1)
        attn_weights_all = torch.cat(attn_weights_all, dim=1)
        return outputs, attn_weights_all


class Seq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        emb_size: int = 1000,
        hidden_size: int = 1000,
        num_layers: int = 4,
        dropout: float = 0.2,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0,
    ):
        super().__init__()
        self.encoder = EncoderRNN(
            vocab_size=src_vocab_size,
            emb_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=src_pad_idx,
        )
        self.decoder = DecoderWithAttention(
            vocab_size=tgt_vocab_size,
            emb_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=tgt_pad_idx,
        )
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != self.src_pad_idx).long()

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_outputs, (h_n, c_n) = self.encoder(src, src_lengths)
        src_mask = self.make_src_mask(src)
        log_probs, attn_weights = self.decoder(
            tgt,
            encoder_outputs,
            src_mask,
            (h_n, c_n),
            teacher_forcing=teacher_forcing,
        )
        return log_probs, attn_weights
