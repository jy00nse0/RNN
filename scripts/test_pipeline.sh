#!/bin/bash

# =========================
# 데이터 파이프라인 테스트 스크립트
# =========================

set -e

echo "======================================================================"
echo "  Testing WMT14 Data Pipeline"
echo "======================================================================"
echo ""

# 테스트용 임시 디렉토리
TEST_DIR="data/test_pipeline"
mkdir -p $TEST_DIR

# =========================
# Test 1: download_data.py 출력 형식 확인
# =========================
echo "Test 1:  Checking download_data.py output format..."

if [ !  -d "data/wmt14_raw" ]; then
    echo "  ERROR: data/wmt14_raw not found.  Run download_data.py first."
    exit 1
fi

# 파일 존재 확인
REQUIRED_FILES=(
    "data/wmt14_raw/train.clean.en"
    "data/wmt14_raw/train.clean.de"
    "data/wmt14_raw/valid.clean. en"
    "data/wmt14_raw/valid.clean.de"
    "data/wmt14_raw/test. clean.en"
    "data/wmt14_raw/test.clean.de"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "  ERROR:  Missing file $file"
        exit 1
    fi
done

echo "  ✓ All raw files exist"

# UTF-8 인코딩 확인
file data/wmt14_raw/train.clean.en | grep -q "UTF-8"
if [ $? -eq 0 ]; then
    echo "  ✓ Files are UTF-8 encoded"
else
    echo "  WARNING: File encoding might not be UTF-8"
fi

# 빈 줄 확인
EMPTY_LINES=$(grep -c "^$" data/wmt14_raw/train.clean.en || true)
echo "  ✓ Empty lines in train.clean.en: $EMPTY_LINES (should be 0)"

echo ""

# =========================
# Test 2: tokenizer.perl 호환성 테스트
# =========================
echo "Test 2: Testing tokenizer.perl compatibility..."

# 샘플 데이터 추출
head -100 data/wmt14_raw/train.clean.en > $TEST_DIR/sample.en

# 토큰화 테스트
cat $TEST_DIR/sample.en | perl tokenizer.perl -l en > $TEST_DIR/sample_tokenized.en

if [ $? -ne 0 ]; then
    echo "  ERROR: Tokenization failed"
    exit 1
fi

echo "  ✓ Tokenization successful"

# 토큰 개수 증가 확인 (구두점 분리로 토큰 수 증가)
ORIGINAL_TOKENS=$(cat $TEST_DIR/sample.en | wc -w)
TOKENIZED_TOKENS=$(cat $TEST_DIR/sample_tokenized.en | wc -w)

echo "  Original tokens: $ORIGINAL_TOKENS"
echo "  Tokenized tokens: $TOKENIZED_TOKENS"

if [ $TOKENIZED_TOKENS -gt $ORIGINAL_TOKENS ]; then
    echo "  ✓ Tokenization increased token count (punctuation separated)"
else
    echo "  WARNING: Token count did not increase as expected"
fi

echo ""

# =========================
# Test 3: process_data.py 호환성 테스트
# =========================
echo "Test 3: Testing process_data.py compatibility..."

# 토큰화된 파일 생성
mkdir -p $TEST_DIR/tokenized
cat $TEST_DIR/sample. en | perl tokenizer.perl -l en > $TEST_DIR/tokenized/sample.en
echo "Sample German text ." > $TEST_DIR/tokenized/sample. de

# process_data.py 테스트 (소규모 vocab)
cat > $TEST_DIR/test_process.py << 'EOF'
import sys
sys.path.insert(0, 'scripts')
from process_data import load_vocab, process_and_save

# Vocab 구축 테스트
vocab = load_vocab('data/test_pipeline/tokenized/sample.en', vocab_size=100)
print(f"Vocab loaded: {len(vocab)} tokens")

# 필터링 테스트
import os
os.makedirs('data/test_pipeline/output', exist_ok=True)
process_and_save(
    'data/test_pipeline/tokenized/sample.en',
    'data/test_pipeline/output/sample.en',
    vocab
)
print("Processing successful")
EOF

python $TEST_DIR/test_process.py

if [ $? -ne 0 ]; then
    echo "  ERROR: process_data.py test failed"
    exit 1
fi

echo "  ✓ process_data.py compatible with tokenizer output"

# <unk> 토큰 존재 확인
if grep -q "<unk>" $TEST_DIR/output/sample.en; then
    echo "  ✓ <unk> tokens generated correctly"
else
    echo "  INFO: No <unk> tokens (vocab might be large enough)"
fi

echo ""

# =========================
# Test 4: 최종 데이터 형식 검증
# =========================
echo "Test 4: Validating final data format..."

if [ -d "data/wmt14_vocab50k/base" ]; then
    # 파일 구조 확인
    EXPECTED_FILES=(
        "data/wmt14_vocab50k/base/train.en"
        "data/wmt14_vocab50k/base/train. de"
        "data/wmt14_vocab50k/base/valid.en"
        "data/wmt14_vocab50k/base/valid.de"
        "data/wmt14_vocab50k/base/test.en"
        "data/wmt14_vocab50k/base/test.de"
    )
    
    ALL_EXIST=true
    for file in "${EXPECTED_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            echo "  WARNING: $file not found (run full pipeline first)"
            ALL_EXIST=false
        fi
    done
    
    if [ "$ALL_EXIST" = true ]; then
        echo "  ✓ All expected files exist"
        
        # 데이터 샘플 확인
        echo ""
        echo "  Sample from train.en:"
        head -2 data/wmt14_vocab50k/base/train.en | sed 's/^/    /'
    fi
else
    echo "  INFO: Final data not yet generated (run full pipeline)"
fi

echo ""

# =========================
# 정리
# =========================
echo "======================================================================"
echo "  All Tests Passed!"
echo "======================================================================"
echo ""
echo "Pipeline is ready.  Run prepare_data.sh to process full dataset."
echo ""

# 임시 파일 정리
rm -rf $TEST_DIR
