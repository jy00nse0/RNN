#!/usr/bin/env python3
import os
import subprocess
import torch

###############################################################################
# 1. Path 설정
###############################################################################

RAW_DATA_DIR = "data/wmt14_vocab50k/base"
SAMPLE_DATA_DIR = "data/sample100k"

TRAIN_SRC = os.path.join(RAW_DATA_DIR, "train.en")
TRAIN_TGT = os.path.join(RAW_DATA_DIR, "train.de")

SAMPLE_SRC = os.path.join(SAMPLE_DATA_DIR, "train.en")
SAMPLE_TGT = os.path.join(SAMPLE_DATA_DIR, "train.de")

VAL_SRC = os.path.join(RAW_DATA_DIR, "valid.en")
VAL_TGT = os.path.join(RAW_DATA_DIR, "valid.de")

TEST_SRC = os.path.join(RAW_DATA_DIR, "test.en")
TEST_TGT = os.path.join(RAW_DATA_DIR, "test.de")


###############################################################################
# 2. 샘플 데이터셋 생성
###############################################################################
def make_sample_dataset(num_lines=100000):
    if not os.path.exists(SAMPLE_DATA_DIR):
        os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)

    print(f"[+] Creating sample dataset using first {num_lines:,} lines...")

    # Create sample train.en
    with open(TRAIN_SRC, "r", encoding="utf-8") as fin, \
         open(SAMPLE_SRC, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= num_lines:
                break
            fout.write(line)

    # Create sample train.de
    with open(TRAIN_TGT, "r", encoding="utf-8") as fin, \
         open(SAMPLE_TGT, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= num_lines:
                break
            fout.write(line)

    # Validation / Test set은 원본 그대로 복사
    os.system(f"cp {VAL_SRC} {SAMPLE_DATA_DIR}/valid.en")
    os.system(f"cp {VAL_TGT} {SAMPLE_DATA_DIR}/valid.de")
    os.system(f"cp {TEST_SRC} {SAMPLE_DATA_DIR}/test.en")
    os.system(f"cp {TEST_TGT} {SAMPLE_DATA_DIR}/test.de")

    print("[+] Sample dataset created successfully.")


###############################################################################
# 3. Sample Test Config
###############################################################################
def run_sample_training():
    print("\n======================= SAMPLE TRAIN START =======================")

    save_path = "checkpoints/sample_test"

    cmd = (
        f"python train.py "
        f"--dataset sample100k "
        f"--save-path {save_path} "
        f"--max-epochs 1 "
        f"--batch-size 64 "
        f"--learning-rate 1.0 "
        f"--encoder-hidden-size 1000 "
        f"--decoder-hidden-size 1000 "
        f"--encoder-num-layers 4 "
        f"--decoder-num-layers 4 "
        f"--attention-type none "
        f"--reverse "
        f"--teacher-forcing-ratio 1.0 "
        f"--cuda"
    )

    print("[Exec]", cmd)
    subprocess.run(cmd, shell=True, check=True)
    print("======================= SAMPLE TRAIN DONE =======================\n")


###############################################################################
# 4. BLEU Evaluation
###############################################################################
def run_sample_evaluation():
    print("\n======================= SAMPLE EVAL START =======================")

    save_path = "checkpoints/sample_test"
    ref_path = "data/sample100k/test.de"

    cmd = (
        f"python calculate_bleu.py "
        f"--model-path {save_path} "
        f"--reference-path {ref_path} "
        f"--epoch 1 "
        f"--cuda"
    )

    print("[Exec]", cmd)
    subprocess.run(cmd, shell=True, check=True)

    print("======================= SAMPLE EVAL DONE ========================\n")


###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    make_sample_dataset()
    run_sample_training()
    run_sample_evaluation()
