import os
import torch
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset
from util import Metadata

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
    
    # 1. Base(Forward) vs Reversed(Backward Source) 결정
    # T1_Base 등 'reverse': False인 실험은 'wmt14-base' 등의 이름을 사용해야 함
    # sample100k는 바로 root_dir을 사용
    if 'sample100k' in args.dataset.lower():
        data_dir = root_dir
    else:
        use_base = 'base' in args.dataset.lower()
        data_dir = os.path.join(root_dir, 'base' if use_base else 'reversed')
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # 2. 방향 결정 (En->De vs De->En)
    # Table 3 실험용
    if 'deen' in args.dataset.lower():
        exts = ('.de', '.en') # Src: German, Tgt: English
        print("Direction: German -> English")
    else:
        exts = ('.en', '.de') # Src: English, Tgt: German
        print("Direction: English -> German")

    # 3. Field 정의
    SRC = Field(tokenize=lambda x: x.split(), init_token='<sos>', eos_token='<eos>', lower=False, batch_first=False)
    TGT = Field(tokenize=lambda x: x.split(), init_token='<sos>', eos_token='<eos>', lower=False, batch_first=False)

    # 4. 데이터셋 로드
    train_data, val_data, test_data = TranslationDataset.splits(
        path=data_dir,
        train='train',
        validation='valid',
        test='test',
        exts=exts,
        fields=(SRC, TGT)
    )

    print(f"Building vocab...")
    SRC.build_vocab(train_data)
    TGT.build_vocab(train_data)
    print(f"Vocab size: SRC={len(SRC.vocab)}, TGT={len(TGT.vocab)}")

    # 5. Iterators
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train_data, val_data, test_data),
        batch_size=args.batch_size,
        device=device,
        sort_key=lambda x: len(x.src),
        sort_within_batch=True
    )

    metadata = Metadata(vocab_size=len(TGT.vocab), padding_idx=TGT.vocab.stoi['<pad>'], vectors=None)

    return metadata, TGT.vocab, BatchWrapper(train_iter), BatchWrapper(val_iter), BatchWrapper(test_iter)

class BatchWrapper:
    def __init__(self, iterator):
        self.iterator = iterator
    def __iter__(self):
        for batch in self.iterator:
            batch.question = batch.src
            batch.answer = batch.trg
            yield batch
    def __len__(self):
        return len(self.iterator)

# Field 생성을 위한 factory (기존 코드 호환용, 필요시 사용)
def field_factory(args):
    return Field(tokenize=lambda x: x.split(), init_token='<sos>', eos_token='<eos>', lower=False)

def metadata_factory(args, vocab):
    return Metadata(vocab_size=len(vocab), padding_idx=vocab.stoi['<pad>'], vectors=None)
