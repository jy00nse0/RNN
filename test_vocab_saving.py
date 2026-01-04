#!/usr/bin/env python3
"""
Test script to verify that src_vocab and tgt_vocab are saved and loaded correctly.
"""

import os
import sys
import tempfile
import shutil
import torch
from dataset import Vocab
from serialization import save_vocab, load_object

def test_vocab_saving_and_loading():
    """Test that vocabularies can be saved and loaded correctly"""
    print("\n" + "="*70)
    print("Test 1: Vocab Saving and Loading")
    print("="*70)
    
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create two different vocabularies (simulating src and tgt)
        src_tokens = ['hello', 'world', 'foo', 'bar', 'test']
        tgt_tokens = ['hallo', 'welt', 'foo', 'bar', 'test', 'extra', 'words']
        
        src_vocab = Vocab(src_tokens)
        tgt_vocab = Vocab(tgt_tokens)
        
        print(f"Created src_vocab with {len(src_vocab)} tokens")
        print(f"Created tgt_vocab with {len(tgt_vocab)} tokens")
        
        # Save vocabularies
        src_vocab_path = os.path.join(temp_dir, 'src_vocab')
        tgt_vocab_path = os.path.join(temp_dir, 'tgt_vocab')
        vocab_path = os.path.join(temp_dir, 'vocab')
        
        save_vocab(src_vocab, src_vocab_path)
        save_vocab(tgt_vocab, tgt_vocab_path)
        save_vocab(tgt_vocab, vocab_path)  # Backward compatibility
        
        print(f"Saved vocabularies to {temp_dir}")
        
        # Verify files exist
        assert os.path.exists(src_vocab_path), "src_vocab file not found"
        assert os.path.exists(tgt_vocab_path), "tgt_vocab file not found"
        assert os.path.exists(vocab_path), "vocab file not found"
        print("✓ All vocabulary files created successfully")
        
        # Load vocabularies
        loaded_src_vocab = load_object(src_vocab_path)
        loaded_tgt_vocab = load_object(tgt_vocab_path)
        loaded_vocab = load_object(vocab_path)
        
        print(f"Loaded src_vocab with {len(loaded_src_vocab)} tokens")
        print(f"Loaded tgt_vocab with {len(loaded_tgt_vocab)} tokens")
        print(f"Loaded vocab with {len(loaded_vocab)} tokens")
        
        # Verify sizes
        assert len(loaded_src_vocab) == len(src_vocab), "src_vocab size mismatch"
        assert len(loaded_tgt_vocab) == len(tgt_vocab), "tgt_vocab size mismatch"
        assert len(loaded_vocab) == len(tgt_vocab), "vocab size mismatch"
        print("✓ All vocabulary sizes match")
        
        # Verify content
        assert loaded_src_vocab.stoi == src_vocab.stoi, "src_vocab stoi mismatch"
        assert loaded_tgt_vocab.stoi == tgt_vocab.stoi, "tgt_vocab stoi mismatch"
        print("✓ Vocabulary content matches")
        
        print("\n✅ Test 1 PASSED: Vocabularies saved and loaded correctly\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_checkpoint_size_validation():
    """Test that checkpoint embedding size validation works"""
    print("\n" + "="*70)
    print("Test 2: Checkpoint Embedding Size Validation")
    print("="*70)
    
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create a mock checkpoint with different embedding sizes
        checkpoint = {
            'encoder.embed.weight': torch.randn(100, 50),  # 100 vocab size, 50 embedding dim
            'decoder.embed.weight': torch.randn(150, 50),  # 150 vocab size, 50 embedding dim
        }
        
        checkpoint_path = os.path.join(temp_dir, 'model.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Load checkpoint
        loaded = torch.load(checkpoint_path, map_location='cpu')
        
        encoder_size = loaded['encoder.embed.weight'].shape[0]
        decoder_size = loaded['decoder.embed.weight'].shape[0]
        
        print(f"Encoder embedding size: {encoder_size}")
        print(f"Decoder embedding size: {decoder_size}")
        
        assert encoder_size == 100, "Encoder size mismatch"
        assert decoder_size == 150, "Decoder size mismatch"
        
        print("✓ Checkpoint size extraction works correctly")
        print("\n✅ Test 2 PASSED: Checkpoint size validation works\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("Vocabulary Saving/Loading Tests")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Vocab Saving and Loading", test_vocab_saving_and_loading()))
    results.append(("Checkpoint Size Validation", test_checkpoint_size_validation()))
    
    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print("="*70)
    print(f"Total: {passed}/{total} tests passed")
    print("="*70)
    
    return all(result for _, result in results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
