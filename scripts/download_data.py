#!/usr/bin/env python3
# ---------------------------------------------------------
# [수정됨] WMT14 English-German 다운로드 스크립트
# 변경사항:
#  - 소스 문장 뒤집기(reverse) 로직 제거 -> process_data.py로 이관
#  - 여전히 문장 길이 ≤50 필터링은 유지 (데이터 용량 최적화)
# ---------------------------------------------------------

import os
from datasets import load_dataset


def process_dataset(dataset, out_src, out_tgt, max_len=50):
    """
    Hugging Face dataset을 처리하여 파일로 저장
    - 길이 필터링 (max_len 이하)
    - 정방향(Forward) 데이터 그대로 저장
    """
    with open(out_src, "w", encoding="utf-8") as osrc, \
         open(out_tgt, "w", encoding="utf-8") as otgt:
        
        count = 0
        for example in dataset:
            src_text = example['translation']['en']
            tgt_text = example['translation']['de']
            
            # 논문과 동일하게 공백 기준 토큰화
            s_tok = src_text.strip().split()
            t_tok = tgt_text.strip().split()

            if len(s_tok) == 0 or len(t_tok) == 0:
                continue
            # 길이 50 초과 필터링 (논문 조건)
            if len(s_tok) > max_len or len(t_tok) > max_len:
                continue

            # Reverse 로직 삭제: 정방향 그대로 저장
            osrc.write(" ".join(s_tok) + "\n")
            otgt.write(" ".join(t_tok) + "\n")
            count += 1
            
        print(f"  - Saved {count} sentences.")


def main():
    out_dir = "data/wmt14_raw"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading WMT14 en-de dataset from Hugging Face...")
    dataset = load_dataset("wmt14", "de-en", trust_remote_code=True)
    
    train_data = dataset['train']
    valid_data = dataset['validation']
    test_data = dataset['test']
    
    print(f"Original sizes - Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

    print("Applying preprocessing (Length <= 50, Forward Sequence)...")
    
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

    print("Done! Raw forward data saved in:", out_dir)


if __name__ == "__main__":
    main()
