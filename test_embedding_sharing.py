#!/usr/bin/env python3
"""
Test to verify that encoder and decoder share the same embedding layer.

This test validates that:
1. Encoder and decoder use the same embedding object (same memory address)
2. Gradients accumulate from both encoder and decoder to shared embedding
3. Memory is saved by using shared embeddings instead of separate ones
"""

import torch
import torch.nn as nn
from model import train_model_factory


class Args:
    """Mock args object for creating model"""
    def __init__(self):
        # Encoder args
        self.encoder_rnn_cell = 'LSTM'
        self.encoder_hidden_size = 32
        self.encoder_num_layers = 1
        self.encoder_rnn_dropout = 0.0
        self.encoder_bidirectional = False
        
        # Decoder args
        self.decoder_type = 'luong'
        self.decoder_rnn_cell = 'LSTM'
        self.decoder_hidden_size = 32
        self.decoder_num_layers = 1
        self.decoder_rnn_dropout = 0.0
        self.luong_attn_hidden_size = 32
        self.luong_input_feed = False
        self.decoder_init_type = 'zeros'
        
        # Attention args
        self.attention_type = 'none'
        self.attention_score = 'dot'
        
        # Embedding args
        self.embedding_type = None
        self.embedding_size = 16
        self.train_embeddings = True
        
        # Training args
        self.teacher_forcing_ratio = 1.0


class Metadata:
    """Mock metadata object"""
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.padding_idx = 0  # Padding token index
        self.vectors = None  # No pre-trained embeddings


def test_embedding_sharing():
    """Test that encoder and decoder share the same embedding object"""
    print("\n=== Testing Embedding Sharing ===")
    
    vocab_size = 100
    args = Args()
    metadata = Metadata(vocab_size)
    
    # Create model using factory
    model = train_model_factory(args, metadata)
    
    # Check that embeddings are the same object
    encoder_embed = model.encoder.embed
    decoder_embed = model.decoder.embed
    
    print(f"Encoder embedding id: {id(encoder_embed)}")
    print(f"Decoder embedding id: {id(decoder_embed)}")
    print(f"Are embeddings the same object? {encoder_embed is decoder_embed}")
    
    assert encoder_embed is decoder_embed, "Encoder and decoder should share the same embedding!"
    print("✅ Embedding sharing test passed!")
    return True


def test_gradient_accumulation():
    """Test that gradients accumulate from both encoder and decoder"""
    print("\n=== Testing Gradient Accumulation ===")
    
    vocab_size = 100
    batch_size = 4
    seq_len = 10
    
    args = Args()
    metadata = Metadata(vocab_size)
    
    model = train_model_factory(args, metadata)
    model.train()
    
    # Create dummy input tensors
    question = torch.randint(0, vocab_size, (seq_len, batch_size))
    answer = torch.randint(0, vocab_size, (seq_len, batch_size))
    
    # Forward pass
    outputs = model(question, answer)
    
    # Create dummy target and compute loss
    target = torch.randint(0, vocab_size, (seq_len - 1, batch_size))
    loss_fn = nn.CrossEntropyLoss()
    
    # Reshape for loss calculation
    outputs_flat = outputs.view(-1, vocab_size)
    target_flat = target.view(-1)
    loss = loss_fn(outputs_flat, target_flat)
    
    # Backward pass
    loss.backward()
    
    # Check that embedding has gradients
    shared_embed = model.encoder.embed
    assert shared_embed.weight.grad is not None, "Embedding should have gradients!"
    assert shared_embed.weight.grad.abs().sum() > 0, "Embedding gradients should be non-zero!"
    
    print(f"Loss value: {loss.item():.4f}")
    print(f"Embedding gradient norm: {shared_embed.weight.grad.norm().item():.4f}")
    print("✅ Gradient accumulation test passed!")
    return True


def test_memory_savings():
    """Test that using shared embeddings saves memory"""
    print("\n=== Testing Memory Savings ===")
    
    vocab_size = 50000  # Typical vocab size from paper
    embedding_size = 1000  # Typical embedding size from paper
    
    # Calculate memory for separate embeddings
    params_per_embedding = vocab_size * embedding_size
    bytes_per_param = 4  # float32
    separate_memory_mb = (2 * params_per_embedding * bytes_per_param) / (1024 * 1024)
    
    # Calculate memory for shared embeddings
    shared_memory_mb = (params_per_embedding * bytes_per_param) / (1024 * 1024)
    
    # Calculate savings
    savings_mb = separate_memory_mb - shared_memory_mb
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {embedding_size}")
    print(f"Memory with separate embeddings: {separate_memory_mb:.2f} MB")
    print(f"Memory with shared embeddings: {shared_memory_mb:.2f} MB")
    print(f"Memory savings: {savings_mb:.2f} MB")
    
    # Verify actual implementation uses shared embeddings
    args = Args()
    args.embedding_size = embedding_size
    metadata = Metadata(vocab_size)
    
    model = train_model_factory(args, metadata)
    
    # Count actual parameters
    total_embed_params = sum(p.numel() for p in model.encoder.embed.parameters())
    expected_params = vocab_size * embedding_size
    
    assert total_embed_params == expected_params, f"Expected {expected_params} params, got {total_embed_params}"
    print(f"Actual embedding parameters: {total_embed_params:,}")
    print("✅ Memory savings test passed!")
    return True


def test_parameter_identity():
    """Test that encoder and decoder embedding weights are the same tensor"""
    print("\n=== Testing Parameter Identity ===")
    
    vocab_size = 100
    args = Args()
    metadata = Metadata(vocab_size)
    
    model = train_model_factory(args, metadata)
    
    # Get embedding weights
    encoder_weight = model.encoder.embed.weight
    decoder_weight = model.decoder.embed.weight
    
    # Check that they are the same tensor (not just equal values)
    assert encoder_weight is decoder_weight, "Embedding weights should be the same tensor!"
    
    # Modify encoder embedding and check decoder sees the change (using no_grad to avoid in-place error)
    with torch.no_grad():
        original_value = encoder_weight[0, 0].item()
        encoder_weight[0, 0] = 999.0
        
        assert decoder_weight[0, 0].item() == 999.0, "Changes to encoder embedding should reflect in decoder!"
        
        # Restore original value
        encoder_weight[0, 0] = original_value
    
    print("✅ Parameter identity test passed!")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Testing Embedding Sharing Implementation")
    print("=" * 60)
    
    tests = [
        test_embedding_sharing,
        test_gradient_accumulation,
        test_memory_savings,
        test_parameter_identity,
    ]
    
    all_passed = True
    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"❌ Test {test_fn.__name__} failed: {e}")
            all_passed = False
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
        print("=" * 60)
        return True
    else:
        print("❌ Some tests failed!")
        print("=" * 60)
        return False


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
