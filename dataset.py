import os
import torch
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset
from util import Metadata

def dataset_factory(args, device):
    """
    WMT14 데이터셋을 로드하여 Iterators와 Metadata를 반환합니다.
    """
    print(f"Loading WMT14 data...")

    # 1. 데이터 경로 설정
    # process_data.py의 출력 경로
    root_dir = 'data/wmt14_vocab50k'
    
    # [논문 재현 중요 포인트]
    # - Base 실험 (Table 1 첫 줄) -> 'base' 폴더 (Forward)
    # - 그 외 모든 실험 (+Reverse, Attention 등) -> 'reversed' 폴더 (Backward Source)
    # train.py의 args.dataset 값을 통해 구분하거나, 기본적으로 reversed를 사용합니다.
    
    use_base = (args.dataset == 'wmt14-base')
    data_dir = os.path.join(root_dir, 'base' if use_base else 'reversed')
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}. Run process_data.py first.")
    
    print(f"Using data from: {data_dir} ({'Forward/Base' if use_base else 'Reversed/Standard'})")

    # 2. Field 정의
    # 논문은 Case-sensitive BLEU를 사용하므로 lower=False
    # process_data.py에서 이미 토큰화(공백 분리)를 했으므로 split()만 사용
    SRC = Field(tokenize=lambda x: x.split(), 
                init_token='<sos>', 
                eos_token='<eos>', 
                lower=False, 
                batch_first=False) # RNN expects (seq_len, batch)
                
    TGT = Field(tokenize=lambda x: x.split(), 
                init_token='<sos>', 
                eos_token='<eos>', 
                lower=False, 
                batch_first=False)

    # 3. 데이터셋 로드 (train, valid, test)
    # extensions: .en (Source), .de (Target)
    train_data, val_data, test_data = TranslationDataset.splits(
        path=data_dir,
        train='train',
        validation='valid',
        test='test',
        exts=('.en', '.de'),
        fields=(SRC, TGT)
    )

    # 4. Vocab 생성
    # process_data.py에서 이미 Top-50k 필터링을 수행했으므로,
    # 여기서는 데이터에 존재하는 토큰 그대로 Vocab을 빌드하면 됩니다.
    print(f"Building vocab from training data...")
    SRC.build_vocab(train_data)
    TGT.build_vocab(train_data)
    
    print(f"Vocab size: SRC={len(SRC.vocab)}, TGT={len(TGT.vocab)}")

    # 5. Iterator 생성 (BucketIterator로 패딩 최소화)
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train_data, val_data, test_data),
        batch_size=args.batch_size,
        device=device,
        sort_key=lambda x: len(x.src), # Source 길이 기준으로 정렬하여 패딩 최소화
        sort_within_batch=True
    )

    # 6. train.py 호환성을 위한 메타데이터 생성
    # train.py는 metadata.padding_idx를 사용함
    metadata = Metadata(
        vocab_size=len(TGT.vocab),
        padding_idx=TGT.vocab.stoi['<pad>'],
        vectors=None # 임베딩은 train.py가 초기화함
    )

    # 7. Batch Wrapper 적용
    # TranslationDataset은 batch.src, batch.trg를 반환하지만,
    # 기존 train.py는 batch.question, batch.answer를 기대함.
    return metadata, TGT.vocab, BatchWrapper(train_iter), BatchWrapper(val_iter), BatchWrapper(test_iter)


class BatchWrapper:
    """
    TranslationDataset의 배치(src, trg)를 
    기존 코드의 필드명(question, answer)으로 매핑해주는 래퍼
    """
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
