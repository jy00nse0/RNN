# decode.py
# ---------------------------------------------------------
# 이 파일은 학습된 NMT 모델을 이용해 "추론(inference)"을 수행한다.
# - greedy_decode(): 단순 argmax 기반 디코딩
# - beam_search(): beam search 기반 디코딩 (논문과 동일 방식)
#   + unknown-word replacement 트릭(attention 기반)
# ---------------------------------------------------------

import torch
import torch.nn.functional as F


@torch.no_grad()
def greedy_decode(
    model,
    src,
    src_lengths,
    max_len=100,
    bos_idx=1,
    eos_idx=2,
):
    """
    가장 단순한 디코딩 방식.
    시간 단축용 실험, 디버깅용으로 쓰기 좋다.

    src: (1, src_len)
    src_lengths: (1,)
    return: 생성된 토큰 ID list (eos 제외)
    """
    device = src.device
    model.eval()

    # Encoder
    encoder_outputs, (h_n, c_n) = model.encoder(src, src_lengths)
    src_mask = model.make_src_mask(src)

    y_prev = torch.tensor([bos_idx], device=device)
    h_c = (h_n, c_n)
    h_tilde_prev = None

    output_tokens = []

    for _ in range(max_len):
        log_probs, h_c, h_tilde_prev, attn_w = \
            model.decoder.forward_step(
                y_prev, h_c, encoder_outputs, src_mask, h_tilde_prev
            )
        next_token = log_probs.argmax(dim=-1).item()
        if next_token == eos_idx:
            break
        output_tokens.append(next_token)
        y_prev = torch.tensor([next_token], device=device)

    return output_tokens


@torch.no_grad()
def beam_search(
    model,
    src,
    src_lengths,
    beam_size=10,
    max_len=100,
    bos_idx=1,
    eos_idx=2,
    unk_idx=None,  # <unk>의 token ID 지정
):
    """
    논문에 사용된 디코딩 방식(Beam Search).
    - 각 hypothesis에 attention history를 유지
    - 최종 best beam 선택
    - <unk> replacement 적용
    
    src: (1, src_len)
    src_lengths: (1,)
    return: 생성된 토큰 ID list (bos/eos 제거)
    """
    device = src.device
    model.eval()

    # Encoder
    encoder_outputs, (h_n, c_n) = model.encoder(src, src_lengths)
    src_mask = model.make_src_mask(src)
    src_len = src.size(1)

    # beam state: (tokens, log_prob, h_c, h_tilde, attn_history)
    beams = [([bos_idx], 0.0, (h_n, c_n), None, [])]
    completed = []

    for _ in range(max_len):
        new_beams = []

        for tokens, log_prob, h_c, h_tilde_prev, attn_hist in beams:
            # EOS로 끝났으면 완료로 이동
            if tokens[-1] == eos_idx:
                completed.append((tokens, log_prob, attn_hist))
                continue

            y_prev = torch.tensor([tokens[-1]], device=device)

            log_probs, new_h_c, new_h_tilde, attn_w = \
                model.decoder.forward_step(
                    y_prev, h_c, encoder_outputs, src_mask, h_tilde_prev
                )

            # (1, src_len) → (src_len,)
            attn_w = attn_w[0].detach()

            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)

            for k in range(beam_size):
                next_token = topk_ids[0, k].item()
                next_log_prob = log_prob + topk_log_probs[0, k].item()

                new_tokens = tokens + [next_token]
                new_attn_hist = attn_hist + [attn_w]

                new_beams.append(
                    (new_tokens, next_log_prob, new_h_c, new_h_tilde, new_attn_hist)
                )

        if not new_beams and completed:
            break

        # 상위 beam_size만 유지
        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

        # 충분한 complete 확보 시 종료
        if len(completed) >= beam_size:
            break

    # 완료된 것이 있다면 best 선택
    if completed:
        completed = sorted(completed, key=lambda x: x[1], reverse=True)
        best_tokens, best_logp, best_attns = completed[0]
    else:
        best_tokens, best_logp, best_attns = beams[0][0], beams[0][1], beams[0][4]

    # bos/eos 제거 + attention 맞춰 정리
    out_tokens = []
    out_attns = []
    for i, tok in enumerate(best_tokens[1:]):  # bos 제외 후부터
        if tok == eos_idx:
            break
        out_tokens.append(tok)
        if i < len(best_attns):
            out_attns.append(best_attns[i])
        else:
            # fallback
            out_attns.append(torch.zeros(src_len, device=device))

    # ----------------------------
    # unknown-word replacement
    # ----------------------------
    if unk_idx is not None:
        src_ids = src[0]  # (src_len,)
        replaced = []
        for i, tok in enumerate(out_tokens):
            if tok == unk_idx and i < len(out_attns):
                src_pos = out_attns[i].argmax().item()
                repl_tok = int(src_ids[src_pos].item())
                replaced.append(repl_tok)
            else:
                replaced.append(tok)
        out_tokens = replaced

    return out_tokens
