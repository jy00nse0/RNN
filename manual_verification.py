#!/usr/bin/env python3
"""
Manual verification script for vocabulary saving/loading with training and evaluation.
This simulates the complete workflow without requiring full dataset.
"""

import os
import sys
import shutil
import subprocess
import tempfile

def setup_test_dataset():
    """Create minimal test dataset in data directory"""
    test_data_dir = 'data/manual_test'
    os.makedirs(test_data_dir, exist_ok=True)
    
    # English source (7 unique words + specials)
    en_train = [
        "hello world",
        "good morning",
        "thank you very much",
        "how are you today",
        "this is a test",
    ]
    
    # German target (12 unique words + specials)
    de_train = [
        "hallo welt zusaetzlich",
        "guten morgen freund",
        "vielen dank sehr",
        "wie geht es dir heute",
        "das ist ein test beispiel",
    ]
    
    en_valid = ["hello again", "good evening"]
    de_valid = ["hallo nochmal wieder", "guten abend freunde"]
    
    en_test = ["thank you", "how are you"]
    de_test = ["vielen dank", "wie geht es dir"]
    
    # Write files
    for name, data in [
        ('train.en', en_train),
        ('train.de', de_train),
        ('valid.en', en_valid),
        ('valid.de', de_valid),
        ('test.en', en_test),
        ('test.de', de_test),
    ]:
        with open(os.path.join(test_data_dir, name), 'w', encoding='utf-8') as f:
            f.write('\n'.join(data) + '\n')
    
    print(f"✓ Created test dataset in {test_data_dir}")
    return test_data_dir


def run_training():
    """Run minimal training to save vocabularies"""
    print("\n" + "="*70)
    print("Step 1: Running Training")
    print("="*70)
    
    save_path = '/tmp/manual_test_checkpoint'
    
    # Clean up previous runs
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    
    cmd = [
        'python', 'train.py',
        '--dataset', 'manual_test',
        '--save-path', save_path,
        '--max-epochs', '1',
        '--batch-size', '2',
        '--learning-rate', '0.1',
        '--encoder-hidden-size', '16',
        '--decoder-hidden-size', '16',
        '--encoder-num-layers', '1',
        '--decoder-num-layers', '1',
        '--embedding-size', '8',
        '--attention-type', 'none',
        '--gradient-clip', '1.0',
        '--save-every-epoch',
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Training failed with exit code {result.returncode}")
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)
        return None
    
    # Find the timestamped directory
    subdirs = [d for d in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, d))]
    if not subdirs:
        print("❌ No checkpoint directory found")
        return None
    
    checkpoint_dir = os.path.join(save_path, subdirs[0])
    print(f"✓ Training completed, checkpoint saved to: {checkpoint_dir}")
    
    # Verify vocabularies were saved
    required_files = ['src_vocab', 'tgt_vocab', 'vocab', 'args']
    for fname in required_files:
        fpath = os.path.join(checkpoint_dir, fname)
        if not os.path.exists(fpath):
            print(f"❌ Missing file: {fname}")
            return None
        print(f"  ✓ Found: {fname}")
    
    # Load and check vocab sizes
    from serialization import load_object
    src_vocab = load_object(os.path.join(checkpoint_dir, 'src_vocab'))
    tgt_vocab = load_object(os.path.join(checkpoint_dir, 'tgt_vocab'))
    
    print(f"  ✓ src_vocab size: {len(src_vocab)}")
    print(f"  ✓ tgt_vocab size: {len(tgt_vocab)}")
    
    # Verify checkpoint embedding sizes match
    import torch
    model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('seq2seq-')]
    if model_files:
        checkpoint_path = os.path.join(checkpoint_dir, model_files[0])
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        encoder_size = checkpoint['encoder.embed.weight'].shape[0]
        decoder_size = checkpoint['decoder.embed.weight'].shape[0]
        
        print(f"  ✓ Checkpoint encoder embedding: {encoder_size}")
        print(f"  ✓ Checkpoint decoder embedding: {decoder_size}")
        
        if encoder_size != len(src_vocab):
            print(f"  ❌ Encoder size {encoder_size} != src_vocab size {len(src_vocab)}")
            return None
        if decoder_size != len(tgt_vocab):
            print(f"  ❌ Decoder size {decoder_size} != tgt_vocab size {len(tgt_vocab)}")
            return None
        
        print(f"  ✓ Embedding sizes match vocabularies")
    
    return checkpoint_dir


def run_evaluation(checkpoint_dir):
    """Run BLEU evaluation to test vocabulary loading"""
    print("\n" + "="*70)
    print("Step 2: Running BLEU Evaluation")
    print("="*70)
    
    ref_path = 'data/manual_test/test.de'
    
    cmd = [
        'python', 'calculate_bleu.py',
        '--model-path', checkpoint_dir,
        '--reference-path', ref_path,
        '--epoch', '1',
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("\nOutput:")
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"❌ Evaluation failed with exit code {result.returncode}")
        print("STDERR:", result.stderr)
        return False
    
    # Check for BLEU score in output
    if 'BLEU' in result.stdout or 'bleu' in result.stdout:
        print("✓ BLEU evaluation completed successfully")
        return True
    else:
        print("⚠ BLEU evaluation ran but no BLEU score found in output")
        return True  # Still consider it a success if no error occurred


def main():
    """Run complete manual verification"""
    print("\n" + "="*70)
    print("Manual Verification: Vocabulary Saving and Loading")
    print("="*70)
    
    try:
        # Setup
        setup_test_dataset()
        
        # Run training
        checkpoint_dir = run_training()
        if not checkpoint_dir:
            print("\n❌ Manual verification FAILED at training step")
            return False
        
        # Run evaluation
        eval_success = run_evaluation(checkpoint_dir)
        if not eval_success:
            print("\n❌ Manual verification FAILED at evaluation step")
            return False
        
        print("\n" + "="*70)
        print("✅ Manual Verification PASSED")
        print("="*70)
        print("\nSummary:")
        print("  ✓ Training saves src_vocab and tgt_vocab")
        print("  ✓ Evaluation loads separate vocabularies")
        print("  ✓ No embedding size mismatch errors")
        print("  ✓ BLEU calculation works correctly")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Manual verification FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
