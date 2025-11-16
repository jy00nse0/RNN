# evaluate.py
# ---------------------------------------------------------
# 테스트 셋 번역 및 BLEU 평가
# sacreBLEU 를 사용하여 WMT 표준 BLEU 점수 계산
# ---------------------------------------------------------

import torch
from sacrebleu import corpus_bleu
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

    for src, src_lengths, tgt, tgt_lengths in dataloader:
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
