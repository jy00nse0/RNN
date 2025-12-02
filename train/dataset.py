# dataset.py
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Optional
import sentencepiece as spm


class MyNMTDataset(Dataset):
    """
    BPE 인코딩 전 텍스트(.bpe.*) 파일을 읽어와서
    BOS / EOS / PAD 토큰 index를 적용해주는 Dataset.
    """
    def __init__(
        self,
        src_path: str,
        tgt_path: str,
        bpe_model_path: str,
        bos_idx: int = 1,
        eos_idx: int = 2,
        pad_idx: int = 0,
        max_len: int = 50,
        max_lines: Optional[int] = None,
    ):
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len
        self.max_lines = max_lines

        # SentencePiece 모델 로드
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_model_path)

        self.src_data, self.tgt_data = self._read_parallel_files(src_path, tgt_path)

        self.src_vocab_size = self.sp.get_piece_size()
        self.tgt_vocab_size = self.sp.get_piece_size()

    def _read_parallel_files(self, src_path, tgt_path):
        """
        Read source and target files in parallel, filtering pairs together.
        A pair is only kept if BOTH source and target satisfy length constraints.
        """
        src_data = []
        tgt_data = []
        with open(src_path, "r", encoding="utf-8") as src_f, \
             open(tgt_path, "r", encoding="utf-8") as tgt_f:
            src_lines = src_f.readlines()
            tgt_lines = tgt_f.readlines()

        if len(src_lines) != len(tgt_lines):
            raise ValueError(
                f"Source and target files have different line counts: "
                f"{len(src_lines)} vs {len(tgt_lines)}"
            )

        for src_line, tgt_line in zip(src_lines, tgt_lines):
            src_toks = src_line.strip().split()
            tgt_toks = tgt_line.strip().split()
            # Keep pair only if both are non-empty and within max_len
            if 0 < len(src_toks) <= self.max_len and 0 < len(tgt_toks) <= self.max_len:
                src_data.append(src_toks)
                tgt_data.append(tgt_toks)
            if self.max_lines is not None and len(src_data) >= self.max_lines:
                break
        return src_data, tgt_data

    def __len__(self):
        return len(self.src_data)

    def _encode(self, tokens):
        """
        tokens는 이미 BPE로 split된 문자열 토큰 리스트
        => SentencePiece vocabulary의 piece ID로 인코딩
        """
        # Join tokens back to string and encode
        return self.sp.encode(' '.join(tokens), out_type=int)

    def __getitem__(self, idx):
        src_tokens = self.src_data[idx]
        tgt_tokens = self.tgt_data[idx]

        # [bos, src..., eos]
        src_ids = [self.bos_idx] + self._encode(src_tokens) + [self.eos_idx]
        tgt_ids = [self.bos_idx] + self._encode(tgt_tokens) + [self.eos_idx]

        src_ids = torch.tensor(src_ids, dtype=torch.long)
        tgt_ids = torch.tensor(tgt_ids, dtype=torch.long)

        return src_ids, tgt_ids

    @staticmethod
    def collate_fn(batch):
        """
        batch = [(src_ids, tgt_ids), ...]
        - 길이 기준 정렬 => pack_padded_sequence 안정화
        - padding 수행
        """
        # Sort by src length to keep src-tgt pairs together
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        
        src_seqs = [b[0] for b in batch]
        tgt_seqs = [b[1] for b in batch]

        src_lengths = torch.tensor([len(s) for s in src_seqs])
        tgt_lengths = torch.tensor([len(t) for t in tgt_seqs])

        # padding
        src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=0)
        tgt_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=0)

        # return:
        # src_padded: (batch, src_len)
        # src_lengths: (batch,)
        # tgt_padded: (batch, tgt_len)
        # tgt_lengths: (batch,)
        return src_padded, src_lengths, tgt_padded, tgt_lengths
