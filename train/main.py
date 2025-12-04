# main.py
# ---------------------------------------------------------
# 실행 진입점: 학습 + 개발 셋 BLEU 평가
# 사용 예:
#   python main.py --data_dir data/wmt14_bpe --epochs 10 --batch_size 128
# ---------------------------------------------------------

import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import sentencepiece as spm

from dataset import MyNMTDataset
from model import Seq2Seq
from train import train_epoch
from evaluate import translate_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--lr_decay_start", type=int, default=5)
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--eval_beam_size", type=int, default=1,
                        help="Beam size used for evaluation (BLEU). Defaults to 1 to speed up evaluation.")
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--train_5000lines", action="store_true", default=False,
                        help="Enable sample test mode: train on first 5000 line pairs only")

    # vocab setting
    parser.add_argument("--pad_idx", type=int, default=0)
    parser.add_argument("--bos_idx", type=int, default=1)
    parser.add_argument("--eos_idx", type=int, default=2)
    parser.add_argument("--unk_idx", type=int, default=3)

    return parser.parse_args()


def build_dataloaders(args):
    sp = spm.SentencePieceProcessor()
    sp.load(f"{args.data_dir}/bpe.model")

    # Determine max_lines based on train_5000lines flag
    max_lines = 5000 if args.train_5000lines else None

    train_dataset = MyNMTDataset(
        src_path=f"{args.data_dir}/train.bpe.en",
        tgt_path=f"{args.data_dir}/train.bpe.de",
        bpe_model_path=f"{args.data_dir}/bpe.model",
        bos_idx=args.bos_idx,
        eos_idx=args.eos_idx,
        pad_idx=args.pad_idx,
        max_lines=max_lines,
    )
    valid_dataset = MyNMTDataset(
        src_path=f"{args.data_dir}/valid.bpe.en",
        tgt_path=f"{args.data_dir}/valid.bpe.de",
        bpe_model_path=f"{args.data_dir}/bpe.model",
        bos_idx=args.bos_idx,
        eos_idx=args.eos_idx,
        pad_idx=args.pad_idx,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=valid_dataset.collate_fn,
    )

    return train_loader, valid_loader, sp, train_dataset.src_vocab_size


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, valid_loader, sp, vocab_size = build_dataloaders(args)

    # Model
    model = Seq2Seq(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        emb_size=1000,
        hidden_size=1000,
        num_layers=4,
        src_pad_idx=args.pad_idx,
        tgt_pad_idx=args.pad_idx,
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    best_bleu = -1.0

    for epoch in range(1, args.epochs + 1):

        # LR decay after N epochs
        if epoch > args.lr_decay_start:
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] / 2

        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            tgt_pad_idx=args.pad_idx,
            device=device,
        )

        bleu = translate_dataset(
            model=model,
            dataloader=valid_loader,
            sp=sp,
            bos_idx=args.bos_idx,
            eos_idx=args.eos_idx,
            unk_idx=args.unk_idx,
            beam_size=args.eval_beam_size,
            max_len=args.max_len,
            device=device,
        )

        print(f"[Epoch {epoch}] loss={train_loss:.4f}, dev BLEU={bleu.score:.2f}")

        # save best
        if bleu.score > best_bleu:
            best_bleu = bleu.score
            save_path = f"{args.save_dir}/best_epoch{epoch}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  → BEST! saved to {save_path}")

    print(f"Training finished. Best BLEU = {best_bleu:.2f}")


if __name__ == "__main__":
    main()
