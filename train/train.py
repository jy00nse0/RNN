# train.py
import torch
import torch.nn as nn
from typing import Tuple


def compute_loss(
    log_probs: torch.Tensor,  # (batch, tgt_len-1, vocab)
    tgt: torch.Tensor,        # (batch, tgt_len)
    pad_idx: int,
) -> torch.Tensor:
    """
    log_probs: 모델 출력 (log_softmax 이미 적용된 상태)
    tgt:       정답 시퀀스 (bos, y1, ..., yN, eos, pad...)
    - log_probs[:, t-1, :] 는 tgt[:, t] 에 해당하는 토큰을 예측한 것
    """
    vocab_size = log_probs.size(-1)

    # 정답 토큰: 첫 토큰(<bos>) 제외, 나머지 (t=1..)
    tgt_gold = tgt[:, 1:]                    # (batch, tgt_len-1)

    # NLLLoss는 (N, C) vs (N,) 형태 요구
    loss_fn = nn.NLLLoss(ignore_index=pad_idx, reduction="sum")

    loss = loss_fn(
        log_probs.reshape(-1, vocab_size),   # (batch*(tgt_len-1), vocab)
        tgt_gold.reshape(-1),                # (batch*(tgt_len-1),)
    )

    # 평균 loss = 토큰 수로 나눔 (pad 제외)
    num_tokens = (tgt_gold != pad_idx).sum().item()
    if num_tokens > 0:
        loss = loss / num_tokens
    return loss


def train_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    tgt_pad_idx: int,
    device: torch.device,
    clip: float = 5.0,
) -> float:
    """
    한 미니배치에 대해:
      - forward
      - loss 계산
      - backward
      - gradient clipping
      - optimizer step
    실행하고 loss(float)를 반환.

    batch: (src, src_lengths, tgt, tgt_lengths)
      src: (batch, src_len)
      src_lengths: (batch,)
      tgt: (batch, tgt_len)  [<bos> ... <eos> ...]
    """
    model.train()

    src, src_lengths, tgt, tgt_lengths = batch
    src = src.to(device)
    src_lengths = src_lengths.to(device)
    tgt = tgt.to(device)

    optimizer.zero_grad()

    # teacher forcing 사용 (논문 기본)
    log_probs, _ = model(
        src,
        src_lengths,
        tgt,
        teacher_forcing=True,
    )  # log_probs: (batch, tgt_len-1, vocab)

    loss = compute_loss(log_probs, tgt, pad_idx=tgt_pad_idx)

    loss.backward()
    torch.nn.utils.clip_grad_norm_((model.parameters(), max_norm=clip)
                                   if hasattr(torch.nn.utils, "clip_grad_norm_")
                                   else None)

    optimizer.step()

    return float(loss.item())


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    tgt_pad_idx: int,
    device: torch.device,
    clip: float = 5.0,
) -> float:
    """
    하나의 epoch 전체를 학습하고 평균 loss 반환.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        loss = train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            tgt_pad_idx=tgt_pad_idx,
            device=device,
            clip=clip,
        )
        total_loss += loss
        n_batches += 1

    if n_batches == 0:
        return 0.0
    return total_loss / n_batches
