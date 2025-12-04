import os
import argparse
from collections import Counter
from tqdm import tqdm

# 논문 Ground Truth
VOCAB_SIZE = 50000
UNK_TOKEN = '<unk>'

def load_vocab(file_path, vocab_size):
    """
    파일에서 단어 빈도수를 계산하여 상위 vocab_size개의 단어 집합을 반환.
    """
    print(f"Building vocabulary from {file_path}...")
    counter = Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            tokens = line.strip().split()
            counter.update(tokens)
    
    most_common = counter.most_common(vocab_size)
    vocab = set([word for word, count in most_common])
    
    print(f"Vocab size: {len(vocab)} (Top-{vocab_size} frequencies)")
    return vocab

def process_and_save_dual(in_path, out_path_base, out_path_rev, vocab, is_source=True):
    """
    입력 파일(정방향)을 읽어:
    1. Vocab 필터링 (<unk> 처리)
    2. Base 경로에는 정방향 그대로 저장
    3. Reversed 경로에는 역방향으로 뒤집어 저장 (Source인 경우만)
    """
    print(f"Processing {in_path}...")
    
    with open(in_path, 'r', encoding='utf-8') as fin, \
         open(out_path_base, 'w', encoding='utf-8') as f_base, \
         open(out_path_rev, 'w', encoding='utf-8') as f_rev:
        
        unknown_count = 0
        total_count = 0
        
        for line in tqdm(fin):
            tokens = line.strip().split()
            
            # 1. Vocab Filtering (<unk> 처리)
            filtered_tokens = []
            for t in tokens:
                total_count += 1
                if t in vocab:
                    filtered_tokens.append(t)
                else:
                    filtered_tokens.append(UNK_TOKEN)
                    unknown_count += 1
            
            # 2. Base Model용 저장 (Forward)
            f_base.write(" ".join(filtered_tokens) + "\n")
            
            # 3. Reverse Model용 저장 (Backward)
            # 논문에 따라 Source 문장만 뒤집음. Target은 그대로 둠.
            if is_source:
                reversed_tokens = list(reversed(filtered_tokens))
                f_rev.write(" ".join(reversed_tokens) + "\n")
            else:
                # Target 데이터는 Reverse 모델에서도 정방향이어야 함
                f_rev.write(" ".join(filtered_tokens) + "\n")

    print(f"  -> Saved Base (Forward): {out_path_base}")
    print(f"  -> Saved Reversed: {out_path_rev}")
    if total_count > 0:
        print(f"  -> Unknown token ratio: {unknown_count/total_count*100:.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='data/wmt14_raw', help='Input directory with raw forward files')
    parser.add_argument('--out_dir', type=str, default='data/wmt14_vocab50k', help='Output directory')
    args = parser.parse_args()

    # 경로 설정
    base_dir = os.path.join(args.out_dir, 'base')       # Base 모델용 (Forward)
    rev_dir = os.path.join(args.out_dir, 'reversed')    # Reverse 모델용 (Backward Source)
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(rev_dir, exist_ok=True)

    # 파일명 정의
    files = {
        'train': ('train.clean.en', 'train.clean.de'),
        'valid': ('valid.clean.en', 'valid.clean.de'),
        'test':  ('test.clean.en', 'test.clean.de')
    }

    # 1. Vocab 구축 (Train 데이터 기준)
    train_src_path = os.path.join(args.raw_dir, files['train'][0])
    train_tgt_path = os.path.join(args.raw_dir, files['train'][1])
    
    vocab_src = load_vocab(train_src_path, VOCAB_SIZE)
    vocab_tgt = load_vocab(train_tgt_path, VOCAB_SIZE)

    # 2. 데이터 처리 (Base와 Reversed 동시 생성)
    for split, (src_file, tgt_file) in files.items():
        print(f"\nProcessing {split} split...")
        
        # Source Files Process
        process_and_save_dual(
            in_path=os.path.join(args.raw_dir, src_file),
            out_path_base=os.path.join(base_dir, split + '.en'),
            out_path_rev=os.path.join(rev_dir, split + '.en'),
            vocab=vocab_src,
            is_source=True # Source 파일은 Reverse 버전에서 뒤집힘
        )
        
        # Target Files Process
        process_and_save_dual(
            in_path=os.path.join(args.raw_dir, tgt_file),
            out_path_base=os.path.join(base_dir, split + '.de'),
            out_path_rev=os.path.join(rev_dir, split + '.de'),
            vocab=vocab_tgt,
            is_source=False # Target 파일은 뒤집지 않음
        )

    print("\nAll processing done.")
    print(f"Base data saved in: {base_dir}")
    print(f"Reversed data saved in: {rev_dir}")

if __name__ == '__main__':
    main()
