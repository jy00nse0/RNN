import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from util import Metadata


SPECIALS = ['<pad>', '<sos>', '<eos>', '<unk>']


class FixedVocab:
    """
    Vocab loaded from preprocessed data.
    No rebuilding, no Counter.
    """
    def __init__(self, vocab_path):
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.itos.append(line.strip())

        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.pad_idx = self.stoi['<pad>']
        self.unk_idx = self.stoi['<unk>']

    def encode(self, tokens):
        return [self.stoi.get(tok, self.unk_idx) for tok in tokens]

    def __len__(self):
        return len(self.itos)


class TranslationDataset(Dataset):
    def __init__(self, src_path, tgt_path, src_vocab, tgt_vocab):
        self.src = []
        self.tgt = []

        with open(src_path, encoding='utf-8') as f:
            for line in f:
                self.src.append(line.strip().split())

        with open(tgt_path, encoding='utf-8') as f:
            for line in f:
                self.tgt.append(line.strip().split())

        assert len(self.src) == len(self.tgt)

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = ['<sos>'] + self.src[idx] + ['<eos>']
        tgt = ['<sos>'] + self.tgt[idx] + ['<eos>']

        return (
            torch.tensor(self.src_vocab.encode(src), dtype=torch.long),
            torch.tensor(self.tgt_vocab.encode(tgt), dtype=torch.long),
        )


def collate_fn(batch, pad_idx):
    src, tgt = zip(*batch)
    src = pad_sequence(src, padding_value=pad_idx)
    tgt = pad_sequence(tgt, padding_value=pad_idx)
    return src, tgt


def dataset_factory(args, device):
    base_dir = 'data/wmt14_vocab50k/base'
    src_ext, tgt_ext = ('de', 'en') if 'deen' in args.dataset else ('en', 'de')

    # vocab 파일은 process_data.py 단계에서 생성돼 있어야 함
    src_vocab = FixedVocab(os.path.join(base_dir, f'vocab.{src_ext}'))
    tgt_vocab = FixedVocab(os.path.join(base_dir, f'vocab.{tgt_ext}'))

    train = TranslationDataset(
        os.path.join(base_dir, f'train.{src_ext}'),
        os.path.join(base_dir, f'train.{tgt_ext}'),
        src_vocab, tgt_vocab
    )

    val = TranslationDataset(
        os.path.join(base_dir, f'valid.{src_ext}'),
        os.path.join(base_dir, f'valid.{tgt_ext}'),
        src_vocab, tgt_vocab
    )

    test = TranslationDataset(
        os.path.join(base_dir, f'test.{src_ext}'),
        os.path.join(base_dir, f'test.{tgt_ext}'),
        src_vocab, tgt_vocab
    )

    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=args.cuda,
        persistent_workers=True,
        collate_fn=lambda b: collate_fn(b, tgt_vocab.pad_idx)
    )

    train_loader = DataLoader(train, shuffle=True, **loader_args)
    val_loader = DataLoader(val, shuffle=False, **loader_args)
    test_loader = DataLoader(test, shuffle=False, **loader_args)

    metadata = Metadata(
        vocab_size=len(tgt_vocab),
        padding_idx=tgt_vocab.pad_idx,
        vectors=None
    )

    return metadata, tgt_vocab, train_loader, val_loader, test_loader
