import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from util import Metadata

class Vocab:
    """Simple vocabulary class"""
    def __init__(self, tokens, specials=['<pad>', '<sos>', '<eos>', '<unk>']):
        self.specials = specials
        self.itos = specials + list(set(tokens) - set(specials))
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.unk_index = self.stoi['<unk>']
        
    def __len__(self):
        return len(self.itos)
    
    def encode(self, tokens):
        return [self.stoi.get(tok, self.unk_index) for tok in tokens]
    
    def decode(self, indices):
        return [self.itos[idx] for idx in indices]

class TranslationDataset(Dataset):
    """Dataset for translation tasks"""
    def __init__(self, src_file, tgt_file, src_vocab=None, tgt_vocab=None, reverse_src=False):
        self.src_sentences = []
        self.tgt_sentences = []
        
        # Read source file
        with open(src_file, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                if reverse_src:
                    tokens = list(reversed(tokens))
                self.src_sentences.append(tokens)
        
        # Read target file
        with open(tgt_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.tgt_sentences.append(line.strip().split())
        
        assert len(self.src_sentences) == len(self.tgt_sentences)
        
        # Build or use provided vocabularies
        if src_vocab is None:
            all_tokens = [tok for sent in self.src_sentences for tok in sent]
            self.src_vocab = Vocab(all_tokens)
        else:
            self.src_vocab = src_vocab
            
        if tgt_vocab is None:
            all_tokens = [tok for sent in self.tgt_sentences for tok in sent]
            self.tgt_vocab = Vocab(all_tokens)
        else:
            self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src = ['<sos>'] + self.src_sentences[idx] + ['<eos>']
        tgt = ['<sos>'] + self.tgt_sentences[idx] + ['<eos>']
        
        src_indices = torch.tensor(self.src_vocab.encode(src), dtype=torch.long)
        tgt_indices = torch.tensor(self.tgt_vocab.encode(tgt), dtype=torch.long)
        
        return src_indices, tgt_indices

def collate_fn(batch, pad_idx=0):
    """Collate function for DataLoader"""
    src_batch, tgt_batch = zip(*batch)
    
    # Pad sequences to the same length
    src_padded = pad_sequence(src_batch, padding_value=pad_idx, batch_first=False)
    tgt_padded = pad_sequence(tgt_batch, padding_value=pad_idx, batch_first=False)
    
    return src_padded, tgt_padded

def dataset_factory(args, device):
    """
    [Revised] WMT14/15 데이터셋 로더
    - args.dataset이 'base'를 포함하면 정방향(Forward) 데이터 사용
    - args.dataset이 'deen'을 포함하면 De->En 방향으로 확장자(.de, .en) 교체
    - args.dataset이 'wmt15'를 포함하면 WMT15 데이터셋 경로 사용
    - args.dataset이 'sample100k'이면 sample 데이터셋 사용
    """
    print(f"Loading data for {args.dataset}...")

    # Determine dataset version (sample100k, WMT14 or WMT15)
    if 'sample100k' in args.dataset.lower():
        root_dir = 'data/sample100k'
    elif 'wmt15' in args.dataset.lower():
        root_dir = 'data/wmt15_vocab50k/base'
    else:
        root_dir = 'data/wmt14_vocab50k/base'
    
    data_dir = root_dir
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # 2. 방향 결정 (En->De vs De->En)
    # Table 3 실험용
    if 'deen' in args.dataset.lower():
        src_ext, tgt_ext = 'de', 'en'  # Src: German, Tgt: English
        print("Direction: German -> English")
    else:
        src_ext, tgt_ext = 'en', 'de'  # Src: English, Tgt: German
        print("Direction: English -> German")

    # Check if we should reverse source sentences
    reverse_src = getattr(args, 'reverse', False)
    
    # Load training data first to build vocabulary
    train_dataset = TranslationDataset(
        os.path.join(data_dir, f'train.{src_ext}'),
        os.path.join(data_dir, f'train.{tgt_ext}'),
        reverse_src=reverse_src
    )
    
    # Use training vocab for validation and test
    val_dataset = TranslationDataset(
        os.path.join(data_dir, f'valid.{src_ext}'),
        os.path.join(data_dir, f'valid.{tgt_ext}'),
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
        reverse_src=reverse_src
    )
    
    test_dataset = TranslationDataset(
        os.path.join(data_dir, f'test.{src_ext}'),
        os.path.join(data_dir, f'test.{tgt_ext}'),
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
        reverse_src=reverse_src
    )
    
    print(f"Vocab size: SRC={len(train_dataset.src_vocab)}, TGT={len(train_dataset.tgt_vocab)}")
    
    # Create DataLoaders
    pad_idx = train_dataset.tgt_vocab.stoi['<pad>']
    
    train_iter = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx)
    )
    
    val_iter = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx)
    )
    
    test_iter = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx)
    )
    
    metadata = Metadata(vocab_size=len(train_dataset.tgt_vocab), padding_idx=pad_idx, vectors=None)
    
    return metadata, train_dataset.tgt_vocab, BatchWrapper(train_iter), BatchWrapper(val_iter), BatchWrapper(test_iter)

class Batch:
    """Wrapper for batch data"""
    def __init__(self, src, trg, device):
        self.src = src.to(device) if device else src
        self.trg = trg.to(device) if device else trg
        self.question = self.src
        self.answer = self.trg

class BatchWrapper:
    def __init__(self, dataloader, device=None):
        self.dataloader = dataloader
        self.device = device
        
    def __iter__(self):
        for src, trg in self.dataloader:
            yield Batch(src, trg, self.device)
    
    def __len__(self):
        return len(self.dataloader)

# Field 생성을 위한 factory (기존 코드 호환용, 필요시 사용)
def field_factory(args):
    # Return a dummy object that won't be used
    return None

def metadata_factory(args, vocab):
    pad_idx = vocab.stoi.get('<pad>', 0)
    return Metadata(vocab_size=len(vocab), padding_idx=pad_idx, vectors=None)
