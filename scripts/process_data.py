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
        for line in tqdm(f, desc="Reading vocab"):
            tokens = line.strip().split()
            counter. update(tokens)
    
    most_common = counter.most_common(vocab_size)
    vocab = set([word for word, count in most_common])
    
    print(f"Vocab size: {len(vocab)} (Top-{vocab_size} frequencies)")
    return vocab

def process_and_save(in_path, out_path, vocab):
    """
    입력 파일을 읽어: 
    1. Vocab 필터링 (<unk> 처리)
    2. 출력 경로에 저장
    """
    print(f"Processing {in_path}...")
    
    with open(in_path, 'r', encoding='utf-8') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        
        unknown_count = 0
        total_count = 0
        
        for line in tqdm(fin, desc=f"Processing {os.path.basename(in_path)}"):
            tokens = line.strip().split()
            
            # Vocab Filtering (<unk> 처리)
            filtered_tokens = []
            for t in tokens:
                total_count += 1
                if t in vocab:
                    filtered_tokens.append(t)
                else:
                    filtered_tokens.append(UNK_TOKEN)
                    unknown_count += 1
            
            # 저장
            fout.write(" ".join(filtered_tokens) + "\n")

    print(f"  -> Saved:  {out_path}")
    if total_count > 0:
        print(f"  -> Unknown token ratio: {unknown_count/total_count*100:.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='data/wmt14_tokenized', 
                       help='Input directory with tokenized files')
    parser.add_argument('--out_dir', type=str, default='data/wmt14_vocab50k', 
                       help='Output directory')
    parser.add_argument('--src_lang', type=str, default='en', 
                       help='Source language code (en or de)')
    parser.add_argument('--tgt_lang', type=str, default='de', 
                       help='Target language code (en or de)')
    args = parser.parse_args()

    # 경로 설정 (base/ 유지)
    base_dir = os.path.join(args. out_dir, 'base')
    os.makedirs(base_dir, exist_ok=True)

    # 파일명 정의 (토큰화된 파일)
    files = {
        'train': (f'train.{args.src_lang}', f'train.{args.tgt_lang}'),
        'valid': (f'valid.{args.src_lang}', f'valid.{args.tgt_lang}'),
        'test':   (f'test.{args.src_lang}', f'test.{args.tgt_lang}')
    }

    # 1.  Vocab 구축 (Train 데이터 기준)
    print("\n" + "="*70)
    print("Step 1: Building Vocabulary")
    print("="*70)
    
    train_src_path = os.path.join(args.raw_dir, files['train'][0])
    train_tgt_path = os.path. join(args.raw_dir, files['train'][1])
    
    print(f"\nSource language: {args.src_lang}")
    vocab_src = load_vocab(train_src_path, VOCAB_SIZE)
    
    print(f"\nTarget language: {args.tgt_lang}")
    vocab_tgt = load_vocab(train_tgt_path, VOCAB_SIZE)

    # 2. 데이터 처리 (Vocab 필터링)
    print("\n" + "="*70)
    print("Step 2: Filtering Data with Vocabulary")
    print("="*70)
    
    for split, (src_file, tgt_file) in files.items():
        print(f"\n--- Processing {split. upper()} split ---")
        
        # Source Files Process
        process_and_save(
            in_path=os.path.join(args.raw_dir, src_file),
            out_path=os.path.join(base_dir, f'{split}.{args.src_lang}'),
            vocab=vocab_src
        )
        
        # Target Files Process
        process_and_save(
            in_path=os.path.join(args.raw_dir, tgt_file),
            out_path=os.path.join(base_dir, f'{split}.{args.tgt_lang}'),
            vocab=vocab_tgt
        )

    print("\n" + "="*70)
    print("All processing done!")
    print("="*70)
    print(f"Processed data saved in: {base_dir}")
    print("\nOutput structure:")
    print(f"  {base_dir}/")
    for split in files.keys():
        print(f"    ├── {split}. {args.src_lang}")
        print(f"    └── {split}.{args.tgt_lang}")

if __name__ == '__main__':
    main()
