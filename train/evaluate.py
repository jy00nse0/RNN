# evaluate.py
# ---------------------------------------------------------
# 테스트 셋 번역 및 BLEU 평가
# sacreBLEU 를 사용하여 WMT 표준 BLEU 점수 계산
# ---------------------------------------------------------

import sys
import os
import torch
from sacrebleu import corpus_bleu

# tqdm 옵션: 설치되어 있으면 진행바를 사용, 아니면 None으로 둬서 기존 동작 유지
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Add test directory to path for decode import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'test'))
from decode import beam_search


@torch.no_grad()
def translate_dataset(
    model,
    dataloader,
    sp,
    bos_idx,
    eos_idx,
    unk_idx=None,
    beam_size=10,
    max_len=100,
    device="cuda",
):
    """
    dataloader: batch_size=1 형태를 가정
    return: 전체 문장 BLEU score
    """
    model.eval()

    all_hypotheses = []
    all_references = []

    # tqdm가 available 하면 진행바로 감싸고, 아니면 원래 dataloader 사용
    iterator = tqdm(dataloader, desc="Translating", unit="sent") if tqdm is not None else dataloader

    for src, src_lengths, tgt, tgt_lengths in iterator:
        # single sentence
        src = src[0:1].to(device)
        src_lengths = src_lengths[0:1].to(device)
        tgt = tgt[0:1].to(device)

        # beam search translation (list[int])
        hyp_ids = beam_search(
            model,
            src,
            src_lengths,
            beam_size=beam_size,
            max_len=max_len,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            unk_idx=unk_idx,
        )

        # decode ids → string
        hyp = sp.decode(hyp_ids)

        # reference sentence ids
        ref_ids = tgt[0].tolist()
        if eos_idx in ref_ids:
            ref_ids = ref_ids[1:ref_ids.index(eos_idx)]
        else:
            ref_ids = ref_ids[1:]
        ref = sp.decode(ref_ids)

        all_hypotheses.append(hyp)
        all_references.append([ref])  # BLEU: list of lists

    bleu = corpus_bleu(all_hypotheses, all_references)
    return bleu
