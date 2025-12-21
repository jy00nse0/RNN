#!/usr/bin/env python3
"""
Integration test demonstrating embedding sharing in a realistic training scenario.

This test shows that the embedding sharing works correctly when:
1. Creating a model using the factory
2. Running a forward pass
3. Computing gradients
4. Verifying that embeddings are truly shared
"""

import torch
import torch.nn as nn
from model import train_model_factory


def main():
    print("=" * 70)
    print("Integration Test: Embedding Sharing in Training Scenario")
    print("=" * 70)
    
    # Configuration matching the problem statement example
    class Args:
        def __init__(self):
            # Encoder configuration
            self.encoder_rnn_cell = 'LSTM'
            self.encoder_hidden_size = 256
            self.encoder_num_layers = 2
            self.encoder_rnn_dropout = 0.2
            self.encoder_bidirectional = False
            
            # Decoder configuration
            self.decoder_type = 'luong'
            self.decoder_rnn_cell = 'LSTM'
            self.decoder_hidden_size = 256
            self.decoder_num_layers = 2
            self.decoder_rnn_dropout = 0.2
            self.luong_attn_hidden_size = 256
            self.luong_input_feed = False
            self.decoder_init_type = 'zeros'
            
            # Attention configuration
            self.attention_type = 'none'
            self.attention_score = 'dot'
            
            # Embedding configuration (matching paper settings)
            self.embedding_type = None
            self.embedding_size = 512  # Reduced from 1000 for this demo
            self.train_embeddings = True
            
            # Training configuration
            self.teacher_forcing_ratio = 0.5
    
    class Metadata:
        def __init__(self):
            self.vocab_size = 10000  # Reduced from 50k for this demo
            self.padding_idx = 0
            self.vectors = None
    
    args = Args()
    metadata = Metadata()
    
    print(f"\nConfiguration:")
    print(f"  Vocabulary size: {metadata.vocab_size:,}")
    print(f"  Embedding dimension: {args.embedding_size}")
    print(f"  Encoder hidden size: {args.encoder_hidden_size}")
    print(f"  Decoder hidden size: {args.decoder_hidden_size}")
    
    # Create model with shared embeddings
    print("\n1. Creating model with train_model_factory()...")
    model = train_model_factory(args, metadata)
    
    # Verify embeddings are shared
    print("\n2. Verifying embedding sharing...")
    encoder_embed = model.encoder.embed
    decoder_embed = model.decoder.embed
    
    print(f"   Encoder embedding address: {id(encoder_embed)}")
    print(f"   Decoder embedding address: {id(decoder_embed)}")
    print(f"   Same object? {encoder_embed is decoder_embed}")
    
    if encoder_embed is not decoder_embed:
        print("   ❌ FAILED: Embeddings are not shared!")
        return False
    print("   ✅ SUCCESS: Embeddings are shared!")
    
    # Calculate memory savings
    print("\n3. Calculating memory savings...")
    params_per_embedding = metadata.vocab_size * args.embedding_size
    bytes_per_param = 4  # float32
    separate_memory_mb = (2 * params_per_embedding * bytes_per_param) / (1024 * 1024)
    shared_memory_mb = (params_per_embedding * bytes_per_param) / (1024 * 1024)
    savings_mb = separate_memory_mb - shared_memory_mb
    
    print(f"   Without sharing: {separate_memory_mb:.2f} MB (2 separate embeddings)")
    print(f"   With sharing: {shared_memory_mb:.2f} MB (1 shared embedding)")
    print(f"   Memory saved: {savings_mb:.2f} MB ({savings_mb/separate_memory_mb*100:.1f}%)")
    
    # Simulate training step
    print("\n4. Simulating training step...")
    model.train()
    
    # Create sample batch
    batch_size = 8
    seq_len = 20
    question = torch.randint(0, metadata.vocab_size, (seq_len, batch_size))
    answer = torch.randint(0, metadata.vocab_size, (seq_len, batch_size))
    
    # Forward pass
    print("   Running forward pass...")
    outputs = model(question, answer)
    print(f"   Output shape: {tuple(outputs.shape)}")
    
    # Compute loss
    target = torch.randint(0, metadata.vocab_size, (seq_len - 1, batch_size))
    loss_fn = nn.CrossEntropyLoss()
    outputs_flat = outputs.view(-1, metadata.vocab_size)
    target_flat = target.view(-1)
    loss = loss_fn(outputs_flat, target_flat)
    
    print(f"   Loss: {loss.item():.4f}")
    
    # Backward pass
    print("   Running backward pass...")
    loss.backward()
    
    # Verify gradients accumulated on shared embedding
    print("\n5. Verifying gradient accumulation...")
    if encoder_embed.weight.grad is None:
        print("   ❌ FAILED: No gradients on embedding!")
        return False
    
    grad_norm = encoder_embed.weight.grad.norm().item()
    grad_nonzero = (encoder_embed.weight.grad.abs() > 0).sum().item()
    total_params = encoder_embed.weight.numel()
    
    print(f"   Gradient norm: {grad_norm:.4f}")
    print(f"   Non-zero gradients: {grad_nonzero:,} / {total_params:,} ({grad_nonzero/total_params*100:.1f}%)")
    print("   ✅ SUCCESS: Gradients accumulated on shared embedding!")
    
    # Verify that encoder and decoder see the same gradients
    print("\n6. Verifying gradient identity...")
    if encoder_embed.weight.grad is not decoder_embed.weight.grad:
        print("   ❌ FAILED: Encoder and decoder have different gradient tensors!")
        return False
    print("   ✅ SUCCESS: Encoder and decoder share the same gradient tensor!")
    
    print("\n" + "=" * 70)
    print("✅ All integration tests passed!")
    print("=" * 70)
    print("\nSummary:")
    print(f"  ✅ Encoder and Decoder share the same embedding parameters")
    print(f"  ✅ Gradients accumulate from both sides (more stable training)")
    print(f"  ✅ Memory saved: {savings_mb:.2f} MB")
    print(f"\nFor full-scale model (50k vocab × 1000 dim):")
    full_scale_savings = (50000 * 1000 * 4) / (1024 * 1024)
    print(f"  Memory savings: ~{full_scale_savings:.0f} MB")
    print("=" * 70)
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
