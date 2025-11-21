#!/usr/bin/env python3
"""
Verification script to check data consistency after download_data.py
Run this after executing download_data.py to verify:
1. All expected files exist
2. Source and target files have matching line counts
3. No empty lines in the data
"""

import os
import sys


def check_file_exists(path):
    """Check if file exists and return its line count"""
    if not os.path.exists(path):
        print(f"❌ Missing: {path}")
        return None
    
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        line_count = len(lines)
        
    # Check for empty lines
    empty_count = sum(1 for line in lines if not line.strip())
    if empty_count > 0:
        print(f"⚠️  Warning: {path} has {empty_count} empty lines")
    
    print(f"✓ {path}: {line_count} lines")
    return line_count


def verify_parallel_data(src_path, tgt_path):
    """Verify that parallel data files have the same number of lines"""
    src_count = check_file_exists(src_path)
    tgt_count = check_file_exists(tgt_path)
    
    if src_count is None or tgt_count is None:
        return False
    
    if src_count != tgt_count:
        print(f"❌ Line count mismatch: {src_path} ({src_count}) vs {tgt_path} ({tgt_count})")
        return False
    
    print(f"✓ Parallel data aligned: {src_count} sentence pairs\n")
    return True


def main():
    data_dir = "data/wmt14_raw"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("   Please run: python scripts/download_data.py")
        sys.exit(1)
    
    print("=" * 60)
    print("Data Verification Report")
    print("=" * 60)
    
    success = True
    
    # Check training data
    print("\n📊 Training data:")
    if not verify_parallel_data(
        os.path.join(data_dir, "train.clean.en"),
        os.path.join(data_dir, "train.clean.de")
    ):
        success = False
    
    # Check validation data
    print("📊 Validation data:")
    if not verify_parallel_data(
        os.path.join(data_dir, "valid.clean.en"),
        os.path.join(data_dir, "valid.clean.de")
    ):
        success = False
    
    # Check test data
    print("📊 Test data:")
    if not verify_parallel_data(
        os.path.join(data_dir, "test.clean.en"),
        os.path.join(data_dir, "test.clean.de")
    ):
        success = False
    
    print("=" * 60)
    if success:
        print("✅ All data files are valid and aligned!")
        print("\nNext steps:")
        print("  1. Run: python scripts/build_bpe.py")
        print("  2. Then: python train/main.py --data_dir data/wmt14_bpe --epochs 10 --batch_size 128")
    else:
        print("❌ Data verification failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
