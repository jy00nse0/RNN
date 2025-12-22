#!/bin/bash

set -e  # 에러 발생 시 중단

# =========================
# WMT14 데이터 전처리 파이프라인
# =========================

echo "======================================================================"
echo "  WMT14 English-German Translation Data Preparation Pipeline"
echo "======================================================================"
echo ""

# 시작 시간 기록
START_TIME=$(date +%s)

# =========================
# Step 1: Raw 데이터 다운로드
# =========================
echo "Step 1/3: Downloading WMT14 dataset from Hugging Face..."
echo "----------------------------------------------------------------------"
python scripts/download_data.py

if [ $? -ne 0 ]; then
    echo "Error: Failed to download data"
    exit 1
fi

echo ""
echo "✓ Download complete"
echo ""

# =========================
# Step 2: 토큰화 (tokenizer.perl)
# =========================
echo "Step 2/3: Tokenizing with Moses tokenizer..."
echo "----------------------------------------------------------------------"

# 디렉토리 생성
mkdir -p data/wmt14_tokenized

# Perl 스크립트 존재 확인
if [ ! -f "tokenizer.perl" ]; then
    echo "Error: tokenizer. perl not found in current directory"
    exit 1
fi

# 영어 토큰화
echo "  [2. 1] Tokenizing English..."
for split in train valid test; do
    echo "    - Processing ${split}. en..."
    cat data/wmt14_raw/${split}.clean.en | \
        perl tokenizer.perl -l en -q -threads 4 > \
        data/wmt14_tokenized/${split}.en
done

# 독일어 토큰화
echo "  [2.2] Tokenizing German..."
for split in train valid test; do
    echo "    - Processing ${split}.de..."
    cat data/wmt14_raw/${split}.clean.de | \
        perl tokenizer. perl -l de -q -threads 4 > \
        data/wmt14_tokenized/${split}.de
done

echo ""
echo "✓ Tokenization complete"
echo ""

# =========================
# Step 3:  Vocabulary 필터링
# =========================
echo "Step 3/3: Building vocabulary (Top 50K) and filtering..."
echo "----------------------------------------------------------------------"

python scripts/process_data.py \
    --raw_dir data/wmt14_tokenized \
    --out_dir data/wmt14_vocab50k \
    --src_lang en \
    --tgt_lang de

if [ $? -ne 0 ]; then
    echo "Error: Failed to process data"
    exit 1
fi

echo ""
echo "✓ Vocabulary filtering complete"
echo ""

# =========================
# 최종 확인
# =========================
echo "======================================================================"
echo "  Data Preparation Complete!"
echo "======================================================================"
echo ""

# 파일 통계 출력
echo "Generated files:"
echo "----------------------------------------------------------------------"
for split in train valid test; do
    EN_FILE="data/wmt14_vocab50k/base/${split}.en"
    DE_FILE="data/wmt14_vocab50k/base/${split}. de"
    
    if [ -f "$EN_FILE" ]; then
        EN_LINES=$(wc -l < "$EN_FILE")
        echo "  ${split}.en: ${EN_LINES} sentences"
    fi
    
    if [ -f "$DE_FILE" ]; then
        DE_LINES=$(wc -l < "$DE_FILE")
        echo "  ${split}.de: ${DE_LINES} sentences"
    fi
done

echo ""

# 샘플 데이터 출력
echo "Sample data (first 3 lines of train. en):"
echo "----------------------------------------------------------------------"
head -3 data/wmt14_vocab50k/base/train.en
echo ""

# 소요 시간 계산
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo "Total time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Next step: Run training with"
echo "  python train.py --dataset wmt14-en-de --cuda"
echo ""
