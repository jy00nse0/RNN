#!/usr/bin/env python3
# ---------------------------------------------------------
# WMT14 English-German 병렬 데이터 자동 다운로드 스크립트
# 논문 전처리 재현:
#  - 문장 길이 ≤50
#  - 소스 문장 뒤집기(reverse)
# ---------------------------------------------------------

import os
import urllib.request
import tarfile
import random


DATA_URL = "https://nlp.stanford.edu/projects/nmt/wmt14.en-de.train.tgz"
DEV_URL = "https://nlp.stanford.edu/projects/nmt/newstest2013.en-de.tgz"
TEST_URL = "https://nlp.stanford.edu/projects/nmt/newstest2014.en-de.tgz"

def download_and_extract(url, dest):
    fname = url.split("/")[-1]
    fpath = os.path.join(dest, fname)

    if not os.path.exists(fpath):
        print(f"Downloading {fname} ...")
        urllib.request.urlretrieve(url, fpath)

    print(f"Extracting {fname} ...")
    with tarfile.open(fpath, "r:gz") as tar:
        tar.extractall(dest)


def process_parallel(src_path, tgt_path, out_src, out_tgt, max_len=50, reverse=True):
    with open(src_path, encoding="utf-8") as fs, \
         open(tgt_path, encoding="utf-8") as ft, \
         open(out_src, "w", encoding="utf-8") as osrc, \
         open(out_tgt, "w", encoding="utf-8") as otgt:

        for s, t in zip(fs, ft):
            s_tok = s.strip().split()
            t_tok = t.strip().split()

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

    # 1) Download raw corpora
    download_and_extract(DATA_URL, out_dir)
    download_and_extract(DEV_URL, out_dir)
    download_and_extract(TEST_URL, out_dir)

    # Extracted file names (Stanford NMT processed version)
    train_src = os.path.join(out_dir, "train.en")
    train_tgt = os.path.join(out_dir, "train.de")
    dev_src = os.path.join(out_dir, "newstest2013.en")
    dev_tgt = os.path.join(out_dir, "newstest2013.de")
    test_src = os.path.join(out_dir, "newstest2014.en")
    test_tgt = os.path.join(out_dir, "newstest2014.de")

    # Ensure existence
    for p in [train_src, train_tgt, dev_src, dev_tgt, test_src, test_tgt]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    # 2) Clean + length≤50 + reverse src
    print("Applying preprocessing...")
    process_parallel(train_src, train_tgt,
                     os.path.join(out_dir, "train.clean.en"),
                     os.path.join(out_dir, "train.clean.de"))
    process_parallel(dev_src, dev_tgt,
                     os.path.join(out_dir, "valid.clean.en"),
                     os.path.join(out_dir, "valid.clean.de"))
    process_parallel(test_src, test_tgt,
                     os.path.join(out_dir, "test.clean.en"),
                     os.path.join(out_dir, "test.clean.de"))

    print("Done! Preprocessed raw data saved in:", out_dir)


if __name__ == "__main__":
    main()
