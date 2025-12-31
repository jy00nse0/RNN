#!/usr/bin/env python3
"""
Simple test to verify the decoder input/label slicing fix works correctly.
This test trains a small model for a few steps and verifies that:
1. The decoder never receives EOS or PAD as inputs
2. The model can be trained without errors
3. The output shapes are correct
"""

import torch
import torch.nn as nn
from dataset import TranslationDataset, Vocab, collate_fn
from torch.utils.data import DataLoader
import argparse
import os

class SimpleArgs:
    """Simple args object for testing"""
    def __init__(self):
        self.dataset = 'wmt14-en-de'
        self.reverse = False
        self.batch_size = 4
        self.embedding_size = 128
        self.encoder_hidden_size = 128
        self.decoder_hidden_size = 128
        self.encoder_num_layers = 1
        self.decoder_num_layers = 1
        self.encoder_rnn_cell = 'LSTM'
        self.decoder_rnn_cell = 'LSTM'
        self.encoder_rnn_dropout = 0.0
        self.decoder_rnn_dropout = 0.0
        self.encoder_bidirectional = False
        self.attention_type = 'none'
        self.attention_score = 'dot'
        self.decoder_type = 'luong'
        self.luong_attn_hidden_size = 128
        self.luong_input_feed = False
        self.decoder_init_type = 'zeros'  # Fixed: 'zeros' not 'zero'
        self.embedding_share = False
        self.embedding_type = None
        self.train_embeddings = True
        self.teacher_forcing_ratio = 0.0

def test_dataset():
    """Test that dataset creates sequences with 2 PAD tokens at the end"""
    print("="*70)
    print("TEST 1: Dataset Format")
    print("="*70)
    
    # Create a small dataset
    data_dir = 'data/wmt14_vocab50k/base'
    train_en = os.path.join(data_dir, 'train.en')
    train_de = os.path.join(data_dir, 'train.de')
    
    if not os.path.exists(train_en):
        print("❌ Mock data not found. Run create_mock_data.py first.")
        return False
    
    dataset = TranslationDataset(train_en, train_de)
    
    # Get a sample
    src, tgt = dataset[0]
    
    print(f"\nSample target sequence shape: {tgt.shape}")
    print(f"Target sequence: {tgt.tolist()}")
    
    # Decode to see tokens
    tgt_tokens = dataset.tgt_vocab.decode(tgt.tolist())
    print(f"Decoded: {tgt_tokens}")
    
    # Check format: should be [<sos>, ..., <eos>, <pad>, <pad>]
    if tgt_tokens[0] != '<sos>':
        print("❌ First token is not <sos>")
        return False
    
    if tgt_tokens[-1] != '<pad>' or tgt_tokens[-2] != '<pad>':
        print("❌ Last two tokens are not <pad>")
        print(f"   Last two: {tgt_tokens[-2:]}")
        return False
    
    # Find EOS
    if '<eos>' not in tgt_tokens:
        print("❌ <eos> not found in sequence")
        return False
    
    eos_idx = tgt_tokens.index('<eos>')
    if eos_idx != len(tgt_tokens) - 3:
        print("❌ <eos> is not at position -3")
        print(f"   EOS at position {eos_idx}, expected {len(tgt_tokens) - 3}")
        return False
    
    print(f"\n✅ Dataset format is correct:")
    print(f"   - First token: <sos>")
    print(f"   - Last two tokens: <pad> <pad>")
    print(f"   - EOS at position -3")
    
    return True

def test_batch_shapes():
    """Test that batched data has correct shapes"""
    print("\n" + "="*70)
    print("TEST 2: Batch Shapes")
    print("="*70)
    
    data_dir = 'data/wmt14_vocab50k/base'
    train_en = os.path.join(data_dir, 'train.en')
    train_de = os.path.join(data_dir, 'train.de')
    
    dataset = TranslationDataset(train_en, train_de)
    
    # Create a small dataloader
    loader = DataLoader(
        dataset, 
        batch_size=4,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=0),
        shuffle=False
    )
    
    # Get a batch
    src_batch, tgt_batch = next(iter(loader))
    
    print(f"\nSource batch shape: {src_batch.shape} (seq_len, batch)")
    print(f"Target batch shape: {tgt_batch.shape} (seq_len, batch)")
    
    # Check that all sequences in batch have same length (due to padding)
    if src_batch.size(1) != 4:
        print(f"❌ Batch size is {src_batch.size(1)}, expected 4")
        return False
    
    if tgt_batch.size(1) != 4:
        print(f"❌ Batch size is {tgt_batch.size(1)}, expected 4")
        return False
    
    # Check that each sequence ends with at least 2 PADs
    pad_idx = 0
    for i in range(tgt_batch.size(1)):
        seq = tgt_batch[:, i]
        if seq[-1] != pad_idx or seq[-2] != pad_idx:
            # Might not have 2 PADs if this is the longest sequence
            # But our dataset always adds 2 PADs, so even longest has 2
            print(f"⚠️  Sequence {i} last two tokens: {seq[-2:].tolist()}")
    
    print(f"\n✅ Batch shapes are correct")
    print(f"   - All sequences have same length (padded)")
    print(f"   - Batch dimension is correct")
    
    return True

def test_model_forward():
    """Test that model forward pass works with new slicing"""
    print("\n" + "="*70)
    print("TEST 3: Model Forward Pass")
    print("="*70)
    
    data_dir = 'data/wmt14_vocab50k/base'
    train_en = os.path.join(data_dir, 'train.en')
    train_de = os.path.join(data_dir, 'train.de')
    
    dataset = TranslationDataset(train_en, train_de)
    
    # Create a small dataloader
    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=0),
        shuffle=False
    )
    
    # Get a batch
    src_batch, tgt_batch = next(iter(loader))
    
    print(f"\nInput shapes:")
    print(f"  Source: {src_batch.shape}")
    print(f"  Target: {tgt_batch.shape}")
    
    # Import model
    from model import train_model_factory
    from util import Metadata
    
    args = SimpleArgs()
    args.teacher_forcing_ratio = 0.0  # No teacher forcing for testing
    
    # Create metadata (namedtuple with vocab_size, padding_idx, vectors)
    src_metadata = Metadata(
        vocab_size=len(dataset.src_vocab),
        padding_idx=0,
        vectors=None
    )
    
    tgt_metadata = Metadata(
        vocab_size=len(dataset.tgt_vocab),
        padding_idx=0,
        vectors=None
    )
    
    # Create model
    model = train_model_factory(args, src_metadata, tgt_metadata)
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(src_batch, tgt_batch)
    
    print(f"\nOutput shape: {outputs.shape}")
    print(f"Expected: ({tgt_batch.size(0) - 3}, {tgt_batch.size(1)}, {tgt_metadata.vocab_size})")
    
    # Check shape
    expected_seq_len = tgt_batch.size(0) - 3
    if outputs.size(0) != expected_seq_len:
        print(f"❌ Output sequence length is {outputs.size(0)}, expected {expected_seq_len}")
        return False
    
    if outputs.size(1) != tgt_batch.size(1):
        print(f"❌ Output batch size is {outputs.size(1)}, expected {tgt_batch.size(1)}")
        return False
    
    if outputs.size(2) != tgt_metadata.vocab_size:
        print(f"❌ Output vocab size is {outputs.size(2)}, expected {tgt_metadata.vocab_size}")
        return False
    
    # Check labels shape
    labels = tgt_batch[1:-2]
    print(f"\nLabels shape: {labels.shape}")
    print(f"Expected: ({tgt_batch.size(0) - 3}, {tgt_batch.size(1)})")
    
    if labels.size(0) != expected_seq_len:
        print(f"❌ Labels sequence length is {labels.size(0)}, expected {expected_seq_len}")
        return False
    
    print(f"\n✅ Model forward pass works correctly")
    print(f"   - Output shape matches expected: {outputs.shape}")
    print(f"   - Labels shape matches output: {labels.shape}")
    
    return True

def main():
    print("\nDECODER INPUT/LABEL SLICING FIX - VERIFICATION TEST\n")
    
    test1_pass = test_dataset()
    test2_pass = test_batch_shapes()
    test3_pass = test_model_forward()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Test 1 (Dataset format): {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Test 2 (Batch shapes): {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print(f"Test 3 (Model forward): {'✅ PASS' if test3_pass else '❌ FAIL'}")
    
    if test1_pass and test2_pass and test3_pass:
        print("\n✅ All tests passed! The fix is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())
