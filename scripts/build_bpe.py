#!/usr/bin/env python3
# ---------------------------------------------------------
# SentencePiece BPE 모델 학습 + train/valid/test에 적용
# ---------------------------------------------------------

import os
import sentencepiece as spm


def train_bpe(input_file, model_prefix, vocab_size=50000):
    """
    SentencePiece BPE 학습
    """
    print(f"Training SentencePiece BPE ... vocab={vocab_size}")
    spm.SentencePieceTrainer.Train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0,
        unk_id=3,
        bos_id=1,
        eos_id=2,
        pad_id=0
    )
    print("BPE model saved:", model_prefix + ".model")


def apply_bpe(sp, in_path, out_path):
    """
    학습된 BPE 모델로 데이터 토큰화
    """
    with open(in_path, encoding="utf-8") as fi, \
         open(out_path, "w", encoding="utf-8") as fo:
        for line in fi:
            out = sp.encode(line.strip(), out_type=str)
            fo.write(" ".join(out) + "\n")


def main():
    raw_dir = "data/wmt14_raw"
    bpe_dir = "data/wmt14_bpe"

    os.makedirs(bpe_dir, exist_ok=True)

    train_src = f"{raw_dir}/train.clean.en"
    train_tgt = f"{raw_dir}/train.clean.de"
    valid_src = f"{raw_dir}/valid.clean.en"
    valid_tgt = f"{raw_dir}/valid.clean.de"
    test_src = f"{raw_dir}/test.clean.en"
    test_tgt = f"{raw_dir}/test.clean.de"

    # 1) SentencePiece 학습
    #    train.en + train.de 모두 기반으로 학습 (일관된 vocab space)
    combined_file = f"{bpe_dir}/spm_input.txt"
    with open(combined_file, "w", encoding="utf-8") as f:
        for p in [train_src, train_tgt]:
            with open(p, encoding="utf-8") as fi:
                for line in fi:
                    f.write(line)

    bpe_model_prefix = f"{bpe_dir}/bpe"
    train_bpe(combined_file, bpe_model_prefix)

    # 2) 토큰화 적용
    sp = spm.SentencePieceProcessor()
    sp.load(bpe_model_prefix + ".model")

    print("Applying BPE to datasets...")
    apply_bpe(sp, train_src, f"{bpe_dir}/train.bpe.en")
    apply_bpe(sp, train_tgt, f"{bpe_dir}/train.bpe.de")
    apply_bpe(sp, valid_src, f"{bpe_dir}/valid.bpe.en")
    apply_bpe(sp, valid_tgt, f"{bpe_dir}/valid.bpe.de")
    apply_bpe(sp, test_src, f"{bpe_dir}/test.bpe.en")
    apply_bpe(sp, test_tgt, f"{bpe_dir}/test.bpe.de")

    print("Done! BPE encoded data saved:", bpe_dir)


if __name__ == "__main__":
    main()
