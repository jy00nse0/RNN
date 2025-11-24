# train.py
import torch
import torch.nn as nn
from typing import Tuple
import time


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
) -> Tuple[float, int]:
    """
    한 미니배치에 대해:
      - forward
      - loss 계산
      - backward
      - gradient clipping
      - optimizer step
    실행하고 loss(float)와 처리된 토큰 수(int)를 반환.

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

    # 처리된 토큰 수 계산 (pad 제외)
    tgt_gold = tgt[:, 1:]
    num_tokens = (tgt_gold != tgt_pad_idx).sum().item()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

    optimizer.step()

    return float(loss.item()), num_tokens


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
    진행률 1% 증가마다 진행률, WPS, ETA를 출력.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    # 전체 토큰 수 계산 (total_words)
    total_words = 0
    for batch in dataloader:
        _, _, tgt, _ = batch
        tgt_gold = tgt[:, 1:]  # 첫 토큰(<bos>) 제외
        num_tokens = (tgt_gold != tgt_pad_idx).sum().item()
        total_words += num_tokens

    # 진행률 추적 변수
    processed_words = 0
    last_reported_progress = -1
    start_time = time.time()

    for batch in dataloader:
        loss, num_tokens = train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            tgt_pad_idx=tgt_pad_idx,
            device=device,
            clip=clip,
        )
        total_loss += loss
        n_batches += 1
        processed_words += num_tokens

        # 진행률 계산 (0~100)
        if total_words > 0:
            progress = (processed_words / total_words) * 100
            current_progress_pct = int(progress)

            # 1% 증가할 때마다 출력
            if current_progress_pct > last_reported_progress:
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    wps = processed_words / elapsed_time  # Words Per Second
                    remaining_words = total_words - processed_words
                    eta_seconds = remaining_words / wps if wps > 0 else 0
                    eta_minutes = eta_seconds / 60

                    print(f"  Progress: {current_progress_pct}% | WPS: {wps:.2f} | ETA: {eta_minutes:.2f} min")
                    last_reported_progress = current_progress_pct

    if n_batches == 0:
        return 0.0
    return total_loss / n_batches
