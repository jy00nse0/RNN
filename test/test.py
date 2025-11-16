# test.py
# ---------------------------------------------------------
# 저장된 checkpoint 로드 → test set 번역 → sacreBLEU 출력
# ---------------------------------------------------------

import argparse
import torch
import sentencepiece as spm
from torch.utils.data import DataLoader

from model import Seq2Seq
from dataset import MyNMTDataset
from evaluate import translate_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--beam_size", type=int, default=10)
    p.add_argument("--max_len", type=int, default=100)

    p.add_argument("--pad_idx", type=int, default=0)
    p.add_argument("--bos_idx", type=int, default=1)
    p.add_argument("--eos_idx", type=int, default=2)
    p.add_argument("--unk_idx", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load(f"{args.data_dir}/bpe.model")

    # Test dataset (WMT newstest2014/2015 선택 가능)
    test_dataset = MyNMTDataset(
        src_path=f"{args.data_dir}/test.bpe.en",
        tgt_path=f"{args.data_dir}/test.bpe.de",
        bpe_model_path=f"{args.data_dir}/bpe.model",
        bos_idx=args.bos_idx,
        eos_idx=args.eos_idx,
        pad_idx=args.pad_idx,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
    )

    vocab_size = sp.get_piece_size()

    # Load model
    model = Seq2Seq(
        vocab_size=vocab_size,
        embed_dim=1000,
        hidden_dim=1000,
        num_layers=4,
        bos_idx=args.bos_idx,
        eos_idx=args.eos_idx,
    ).to(device)

    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    print(f"Loaded checkpoint: {args.ckpt_path}")

    # Calculate BLEU
    bleu = translate_dataset(
        model=model,
        dataloader=test_loader,
        sp=sp,
        bos_idx=args.bos_idx,
        eos_idx=args.eos_idx,
        unk_idx=args.unk_idx,
        beam_size=args.beam_size,
        max_len=args.max_len,
        device=device,
    )

    print(f"[TEST BLEU] {bleu.score:.2f}")


if __name__ == "__main__":
    main()
