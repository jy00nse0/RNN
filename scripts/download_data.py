#!/usr/bin/env python3
# ---------------------------------------------------------
# WMT14 English-German 병렬 데이터 자동 다운로드 스크립트 (Hugging Face datasets 사용)
# 논문 전처리 재현:
#  - 문장 길이 ≤50
#  - 소스 문장 뒤집기(reverse)
# ---------------------------------------------------------

import os
from datasets import load_dataset


def process_dataset(dataset, out_src, out_tgt, max_len=50, reverse=True):
    """
    Hugging Face dataset을 처리하여 파일로 저장
    - 길이 필터링 (max_len 이하)
    - 소스 문장 뒤집기 (reverse=True)
    """
    with open(out_src, "w", encoding="utf-8") as osrc, \
         open(out_tgt, "w", encoding="utf-8") as otgt:
        
        for example in dataset:
            # Hugging Face WMT14 dataset의 'translation' 필드 접근
            src_text = example['translation']['en']
            tgt_text = example['translation']['de']
            
            s_tok = src_text.strip().split()
            t_tok = tgt_text.strip().split()

            if len(s_tok) == 0 or len(t_tok) == 0:
                continue
            if len(s_tok) > max_len or len(t_tok) > max_len:
                continue

            if reverse:
                s_tok = list(reversed(s_tok))

            osrc.write(" ".join(s_tok) + "\n")
            otgt.write(" ".join(t_tok) + "\n")


def main():
    out_dir = "data/wmt14_raw"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading WMT14 en-de dataset from Hugging Face...")
    # Hugging Face datasets에서 WMT14 en-de 데이터 로드
    # Configuration should be "de-en" which means German-English pair in HF WMT14
    dataset = load_dataset("wmt14", "de-en", trust_remote_code=True)
    
    # dataset에는 'train', 'validation', 'test' split이 있음
    train_data = dataset['train']
    valid_data = dataset['validation']  # newstest2013 상당
    test_data = dataset['test']  # newstest2014 상당
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(valid_data)}")
    print(f"Test samples: {len(test_data)}")

    # 전처리 적용 + 파일 저장
    # 출력 파일명은 기존 스크립트와 동일하게 유지
    print("Applying preprocessing (length≤50, reverse source)...")
    
    print("Processing train split...")
    process_dataset(train_data,
                    os.path.join(out_dir, "train.clean.en"),
                    os.path.join(out_dir, "train.clean.de"))
    
    print("Processing validation split...")
    process_dataset(valid_data,
                    os.path.join(out_dir, "valid.clean.en"),
                    os.path.join(out_dir, "valid.clean.de"))
    
    print("Processing test split...")
    process_dataset(test_data,
                    os.path.join(out_dir, "test.clean.en"),
                    os.path.join(out_dir, "test.clean.de"))

    print("Done! Preprocessed data saved in:", out_dir)


if __name__ == "__main__":
    main()
