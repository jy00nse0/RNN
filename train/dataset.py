# dataset.py
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
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
        max_lines: int = None,
    ):
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len
        self.max_lines = max_lines

        # SentencePiece 모델 로드
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_model_path)

        self.src_data = self._read_file(src_path)
        self.tgt_data = self._read_file(tgt_path)

        assert len(self.src_data) == len(self.tgt_data), "src/tgt 라인 수 다름"

        self.src_vocab_size = self.sp.get_piece_size()
        self.tgt_vocab_size = self.sp.get_piece_size()

    def _read_file(self, path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                toks = line.strip().split()
                if 0 < len(toks) <= self.max_len:
                    data.append(toks)
                if self.max_lines is not None and len(data) >= self.max_lines:
                    break
        return data

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
