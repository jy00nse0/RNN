#!/usr/bin/env python3
"""
Test to verify that source sequence reversal happens only once.

This test verifies the fix for the double reversal bug:
- When --reverse flag is set, sequences should be reversed ONCE in dataset.py
- No additional reversal should occur in train.py during training/evaluation

Expected behavior:
1. Dataset-level reversal: tokens are reversed when reading from file
2. No batch-level reversal: train() and evaluate() functions don't reverse
"""

import sys
import os
import tempfile
import torch

# Add the project root to the path
sys.path.insert(0, '/home/runner/work/RNN/RNN')

from dataset import TranslationDataset, Vocab


def test_dataset_reversal():
    """Test that dataset reversal works correctly"""
    print("="*60)
    print("Test 1: Dataset reversal with reverse_src=True")
    print("="*60)
    
    # Create temporary test files
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.src') as f:
        src_file = f.name
        f.write("hello world\n")
        f.write("foo bar baz\n")
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tgt') as f:
        tgt_file = f.name
        f.write("hallo welt\n")
        f.write("fu baz qux\n")
    
    try:
        # Test with reverse_src=True
        dataset_reversed = TranslationDataset(src_file, tgt_file, reverse_src=True)
        
        # Check that tokens are reversed
        # Original: "hello world"
        # Expected: "world hello" (reversed)
        assert dataset_reversed.src_sentences[0] == ['world', 'hello'], \
            f"Expected ['world', 'hello'], got {dataset_reversed.src_sentences[0]}"
        
        # Original: "foo bar baz"
        # Expected: "baz bar foo" (reversed)
        assert dataset_reversed.src_sentences[1] == ['baz', 'bar', 'foo'], \
            f"Expected ['baz', 'bar', 'foo'], got {dataset_reversed.src_sentences[1]}"
        
        print("✓ Dataset reversal works correctly with reverse_src=True")
        
        # Test with reverse_src=False
        dataset_normal = TranslationDataset(src_file, tgt_file, reverse_src=False)
        
        # Check that tokens are NOT reversed
        assert dataset_normal.src_sentences[0] == ['hello', 'world'], \
            f"Expected ['hello', 'world'], got {dataset_normal.src_sentences[0]}"
        
        assert dataset_normal.src_sentences[1] == ['foo', 'bar', 'baz'], \
            f"Expected ['foo', 'bar', 'baz'], got {dataset_normal.src_sentences[1]}"
        
        print("✓ Dataset works correctly with reverse_src=False")
        
    finally:
        # Clean up temporary files
        os.unlink(src_file)
        os.unlink(tgt_file)
    
    return True


def test_no_batch_reverse_function():
    """Test that batch_reverse_source function is removed from train.py"""
    print("\n" + "="*60)
    print("Test 2: Verify batch_reverse_source is removed")
    print("="*60)
    
    # Read train.py and check that batch_reverse_source is not defined
    with open('/home/runner/work/RNN/RNN/train.py', 'r') as f:
        train_content = f.read()
    
    # Check that the function definition doesn't exist
    if 'def batch_reverse_source' in train_content:
        print("✗ batch_reverse_source function still exists in train.py")
        return False
    
    print("✓ batch_reverse_source function removed from train.py")
    
    # Check that reverse_src parameter is not used in train() and evaluate()
    if 'reverse_src=' in train_content:
        print("✗ reverse_src parameter still used in train.py")
        return False
    
    print("✓ reverse_src parameter removed from train() and evaluate() calls")
    
    return True


def test_function_signatures():
    """Test that function signatures are correct"""
    print("\n" + "="*60)
    print("Test 3: Verify function signatures")
    print("="*60)
    
    from train import train, evaluate
    import inspect
    
    # Check train() signature
    train_sig = inspect.signature(train)
    train_params = list(train_sig.parameters.keys())
    
    expected_train_params = ['model', 'optimizer', 'train_iter', 'metadata', 'grad_clip']
    if train_params != expected_train_params:
        print(f"✗ train() signature incorrect: {train_params}")
        print(f"  Expected: {expected_train_params}")
        return False
    
    print(f"✓ train() signature correct: {train_params}")
    
    # Check evaluate() signature
    eval_sig = inspect.signature(evaluate)
    eval_params = list(eval_sig.parameters.keys())
    
    expected_eval_params = ['model', 'val_iter', 'metadata']
    if eval_params != expected_eval_params:
        print(f"✗ evaluate() signature incorrect: {eval_params}")
        print(f"  Expected: {expected_eval_params}")
        return False
    
    print(f"✓ evaluate() signature correct: {eval_params}")
    
    return True


def main():
    """Run all tests"""
    print("Testing Source Sequence Reversal Fix")
    print("="*60)
    
    all_tests_passed = True
    
    # Run tests
    tests = [
        test_dataset_reversal,
        test_no_batch_reverse_function,
        test_function_signatures
    ]
    
    for test in tests:
        try:
            if not test():
                all_tests_passed = False
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            all_tests_passed = False
    
    # Final result
    print("\n" + "="*60)
    if all_tests_passed:
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nSummary:")
        print("- Dataset reversal happens once when reverse_src=True")
        print("- No batch-level reversal in train.py")
        print("- Function signatures are correct")
        print("- Fix successfully eliminates double reversal bug")
        return 0
    else:
        print("SOME TESTS FAILED! ✗")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
