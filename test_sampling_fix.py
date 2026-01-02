#!/usr/bin/env python3
"""
Test to validate that the decoder sampling correctly handles EOS tokens
and doesn't generate repetitive tokens infinitely.
"""

import torch
import torch.nn as nn
from model.seq2seq.sampling import GreedySampler, RandomSampler, BeamSearch


class MockDecoder(nn.Module):
    """Mock decoder that simulates a scenario where it would generate 
    repetitive tokens (like commas) if not stopped properly at EOS."""
    
    def __init__(self, vocab_size, eos_idx, comma_idx):
        super(MockDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.eos_idx = eos_idx
        self.comma_idx = comma_idx
        self.call_count = 0
        
    def forward(self, t, input, encoder_outputs, h_n, **kwargs):
        batch_size = input.size(0)
        
        # Simulate a decoder that:
        # - Generates a few words (steps 0-2)
        # - Then generates EOS at step 3
        # - Would generate commas repeatedly after that if not stopped
        output = torch.zeros(batch_size, self.vocab_size)
        
        if t < 3:
            # Generate some tokens (not comma, not EOS)
            output[:, t + 1] = 10.0  # High probability for token at position t+1
        elif t == 3:
            # Generate EOS
            output[:, self.eos_idx] = 10.0
        else:
            # After EOS, would want to generate comma (the bug scenario)
            output[:, self.comma_idx] = 10.0
            
        self.call_count += 1
        return output, None, kwargs


def test_greedy_sampler_stops_at_eos():
    """Test that GreedySampler stops generating when EOS is encountered."""
    print("Testing GreedySampler...")
    
    vocab_size = 100
    sos_idx = 0
    eos_idx = 50
    comma_idx = 10
    max_length = 20  # Long enough to generate many commas if bug exists
    
    # Create mock encoder outputs and hidden state
    batch_size = 2
    seq_len = 5
    hidden_size = 256
    encoder_outputs = torch.randn(seq_len, batch_size, hidden_size)
    h_n = torch.randn(1, batch_size, hidden_size)
    
    # Create mock decoder
    decoder = MockDecoder(vocab_size, eos_idx, comma_idx)
    
    # Test GreedySampler
    sampler = GreedySampler()
    sequences, lengths = sampler.sample(encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length)
    
    print(f"  Generated sequences shape: {sequences.shape}")
    print(f"  Sequence lengths: {lengths}")
    print(f"  First sequence: {sequences[0].tolist()}")
    
    # Verify that:
    # 1. Sequences contain EOS
    assert (sequences == eos_idx).any(dim=1).all(), "All sequences should contain EOS"
    
    # 2. After EOS, no new tokens should be generated (should all be EOS)
    for b in range(batch_size):
        seq = sequences[b].tolist()
        eos_pos = seq.index(eos_idx)
        # Everything after first EOS should also be EOS (since we keep outputting EOS)
        for i in range(eos_pos + 1, len(seq)):
            assert seq[i] == eos_idx, f"After EOS at position {eos_pos}, all tokens should be EOS, but found {seq[i]} at position {i}"
    
    # 3. Should not contain repeated commas (the bug scenario)
    for b in range(batch_size):
        seq = sequences[b].tolist()
        # Count commas before EOS
        eos_pos = seq.index(eos_idx)
        commas_before_eos = seq[:eos_pos].count(comma_idx)
        assert commas_before_eos == 0, f"Should not have commas before EOS, but found {commas_before_eos}"
    
    # 4. Decoder should not be called excessively (should stop early)
    # It should be called around 4-5 times per batch (until EOS is generated), not 20 times
    max_expected_calls = 6 * batch_size  # Some margin
    assert decoder.call_count <= max_expected_calls, f"Decoder called {decoder.call_count} times, expected <= {max_expected_calls}"
    
    print("  ✓ GreedySampler correctly stops at EOS")
    print()


def test_random_sampler_stops_at_eos():
    """Test that RandomSampler stops generating when EOS is encountered."""
    print("Testing RandomSampler...")
    
    vocab_size = 100
    sos_idx = 0
    eos_idx = 50
    comma_idx = 10
    max_length = 20
    
    batch_size = 2
    seq_len = 5
    hidden_size = 256
    encoder_outputs = torch.randn(seq_len, batch_size, hidden_size)
    h_n = torch.randn(1, batch_size, hidden_size)
    
    decoder = MockDecoder(vocab_size, eos_idx, comma_idx)
    
    sampler = RandomSampler()
    sequences, lengths = sampler.sample(encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length)
    
    print(f"  Generated sequences shape: {sequences.shape}")
    print(f"  Sequence lengths: {lengths}")
    
    # Verify sequences contain EOS
    assert (sequences == eos_idx).any(dim=1).all(), "All sequences should contain EOS"
    
    # Verify decoder wasn't called excessively
    max_expected_calls = 6 * batch_size
    assert decoder.call_count <= max_expected_calls, f"Decoder called {decoder.call_count} times, expected <= {max_expected_calls}"
    
    print("  ✓ RandomSampler correctly stops at EOS")
    print()


def test_beam_search_stops_at_eos():
    """Test that BeamSearch stops generating when EOS is encountered."""
    print("Testing BeamSearch...")
    
    vocab_size = 100
    sos_idx = 0
    eos_idx = 50
    comma_idx = 10
    max_length = 20
    
    batch_size = 2
    seq_len = 5
    hidden_size = 256
    encoder_outputs = torch.randn(seq_len, batch_size, hidden_size)
    h_n = torch.randn(1, batch_size, hidden_size)
    
    decoder = MockDecoder(vocab_size, eos_idx, comma_idx)
    
    sampler = BeamSearch(beam_width=3)
    sequences, lengths = sampler.sample(encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length)
    
    print(f"  Generated sequences shape: {sequences.shape}")
    print(f"  Sequence lengths: {lengths}")
    
    # Verify sequences contain EOS
    assert (sequences == eos_idx).any(dim=1).all(), "All sequences should contain EOS"
    
    print("  ✓ BeamSearch correctly handles EOS")
    print()


def main():
    print("=" * 70)
    print("Testing Decoder Sampling Fix for Repetitive Token Generation")
    print("=" * 70)
    print()
    
    torch.manual_seed(42)  # For reproducibility
    
    test_greedy_sampler_stops_at_eos()
    test_random_sampler_stops_at_eos()
    test_beam_search_stops_at_eos()
    
    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
