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
        debug: bool = True,
    ):
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len
        self.debug = debug

        # SentencePiece 모델 로드
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_model_path)

        self.src_data = self._read_file(src_path, "src")
        self.tgt_data = self._read_file(tgt_path, "tgt")

        # Enhanced assertion with detailed error reporting
        try:
            assert len(self.src_data) == len(self.tgt_data), "src/tgt 라인 수 다름"
        except AssertionError:
            src_len = len(self.src_data)
            tgt_len = len(self.tgt_data)
            print(f"\n[ERROR] Source/Target line count mismatch!")
            print(f"  Source lines: {src_len}")
            print(f"  Target lines: {tgt_len}")
            print(f"  Difference: {abs(src_len - tgt_len)}")
            
            # Show sample mismatches
            min_len = min(src_len, tgt_len)
            
            if min_len > 0:
                print(f"\n  First few aligned samples (up to 5):")
                for i in range(min(5, min_len)):
                    src_sample = ' '.join(self.src_data[i][:10])  # First 10 tokens
                    tgt_sample = ' '.join(self.tgt_data[i][:10])
                    print(f"    [{i}] src: {src_sample}...")
                    print(f"    [{i}] tgt: {tgt_sample}...")
            
            # Show where data is missing
            if src_len != tgt_len:
                print(f"\n  Missing data indices:")
                if src_len > tgt_len:
                    print(f"    Source has {src_len - tgt_len} extra lines")
                    print(f"    Extra indices: {tgt_len} to {src_len - 1}")
                    # Show first few extra samples
                    for i in range(min(5, src_len - tgt_len)):
                        idx = tgt_len + i
                        if idx < src_len:
                            src_sample = ' '.join(self.src_data[idx][:10])
                            print(f"      [{idx}] src: {src_sample}...")
                else:
                    print(f"    Target has {tgt_len - src_len} extra lines")
                    print(f"    Extra indices: {src_len} to {tgt_len - 1}")
                    # Show first few extra samples
                    for i in range(min(5, tgt_len - src_len)):
                        idx = src_len + i
                        if idx < tgt_len:
                            tgt_sample = ' '.join(self.tgt_data[idx][:10])
                            print(f"      [{idx}] tgt: {tgt_sample}...")
            
            raise AssertionError(
                f"Source/Target line count mismatch: src={src_len}, tgt={tgt_len}"
            )

        self.src_vocab_size = self.sp.get_piece_size()
        self.tgt_vocab_size = self.sp.get_piece_size()

    def _read_file(self, path, file_type=""):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                toks = line.strip().split()
                if 0 < len(toks) <= self.max_len:
                    data.append(toks)
        
        # Debug output
        if self.debug:
            print(f"Loaded {file_type}_path: {path} ({len(data)} lines)")
            if len(data) > 0:
                samples = []
                for i in range(min(3, len(data))):
                    samples.append(' '.join(data[i]))
                print(f"  Showing up to 3 samples: {samples}")
        
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
