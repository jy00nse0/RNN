#!/usr/bin/env python3
"""
Test to demonstrate the decoder input/label slicing bug.

This test shows that the current implementation incorrectly uses EOS and PAD
tokens as decoder inputs during training.
"""

import torch

def test_current_implementation():
    """Test showing the bug in the current implementation."""
    print("="*70)
    print("DEMONSTRATING THE BUG IN CURRENT IMPLEMENTATION")
    print("="*70)
    
    # Simulate a padded answer sequence
    # Format: [SOS, tok1, tok2, EOS, PAD, PAD]
    # SOS=0, tok1=5, tok2=6, EOS=1, PAD=2
    answer = torch.tensor([[0], [5], [6], [1], [2], [2]])
    answer_seq_len = answer.size(0)
    
    print(f"\nInput sequence: {answer.squeeze().tolist()}")
    print(f"  SOS=0, tok1=5, tok2=6, EOS=1, PAD=2")
    print(f"Tensor shape: {answer.shape} (seq_len={answer_seq_len}, batch=1)")
    
    print(f"\nCurrent training loop:")
    print(f"  Loop range: range({answer_seq_len} - 1) = range({answer_seq_len - 1}) = {list(range(answer_seq_len - 1))}")
    print(f"\nDecoder steps (with teacher forcing):")
    
    input_word = answer[0]  # Initial input
    has_bug = False
    
    for t in range(answer_seq_len - 1):
        output_idx = t
        label = answer[t + 1]
        
        status = ""
        if input_word.item() == 1:  # EOS
            status += " ⚠️  BUG: Using EOS as input!"
            has_bug = True
        if input_word.item() == 2:  # PAD
            status += " ⚠️  BUG: Using PAD as input!"
            has_bug = True
        if label.item() == 2:  # PAD
            status += " ⚠️  BUG: Predicting PAD!"
            has_bug = True
        
        print(f"  t={t}: input={input_word.item()}, outputs[{output_idx}] -> label={label.item()}{status}")
        
        # Teacher forcing: use next token as input
        input_word = answer[t + 1]
    
    print(f"\nLabels: answer[1:] = {answer[1:].squeeze().tolist()}")
    print(f"Outputs shape: ({answer_seq_len - 1}, batch, vocab_size)")
    
    if has_bug:
        print(f"\n❌ BUG DETECTED: Decoder receives EOS/PAD as inputs!")
    else:
        print(f"\n✓ No bugs detected")
    
    return has_bug


def test_proposed_fix():
    """Test showing the fix."""
    print("\n" + "="*70)
    print("PROPOSED FIX")
    print("="*70)
    
    # Same sequence as before
    answer = torch.tensor([[0], [5], [6], [1], [2], [2]])
    answer_seq_len = answer.size(0)
    
    print(f"\nInput sequence: {answer.squeeze().tolist()}")
    print(f"  SOS=0, tok1=5, tok2=6, EOS=1, PAD=2")
    
    print(f"\nProposed training loop:")
    print(f"  Loop range: range({answer_seq_len} - 2) = range({answer_seq_len - 2}) = {list(range(answer_seq_len - 2))}")
    print(f"\nDecoder steps (with teacher forcing):")
    
    input_word = answer[0]  # Initial input
    has_bug = False
    
    for t in range(answer_seq_len - 2):
        output_idx = t
        label = answer[t + 1]
        
        status = ""
        if input_word.item() == 1:  # EOS
            status += " ⚠️  BUG: Using EOS as input!"
            has_bug = True
        if input_word.item() == 2:  # PAD
            status += " ⚠️  BUG: Using PAD as input!"
            has_bug = True
        if label.item() == 2:  # PAD
            status += " ⚠️  BUG: Predicting PAD!"
            has_bug = True
        
        print(f"  t={t}: input={input_word.item()}, outputs[{output_idx}] -> label={label.item()}{status}")
        
        # Teacher forcing: use next token as input
        input_word = answer[t + 1]
    
    print(f"\nLabels: answer[1:-1] = {answer[1:-1].squeeze().tolist()}")
    print(f"Outputs shape: ({answer_seq_len - 2}, batch, vocab_size)")
    
    if has_bug:
        print(f"\n❌ BUG STILL PRESENT!")
    else:
        print(f"\n✅ FIX SUCCESSFUL: No EOS/PAD used as inputs!")
    
    return not has_bug


def test_unpadded_sequence():
    """Test with a sequence that has no padding (worst case for proposed fix)."""
    print("\n" + "="*70)
    print("TEST WITH UNPADDED SEQUENCE")
    print("="*70)
    
    # Sequence with no padding: [SOS, A, B, EOS]
    answer = torch.tensor([[0], [5], [6], [1]])
    answer_seq_len = answer.size(0)
    
    print(f"\nInput sequence: {answer.squeeze().tolist()}")
    print(f"  SOS=0, A=5, B=6, EOS=1")
    
    print(f"\n--- Current Implementation ---")
    print(f"Loop range: range({answer_seq_len - 1}) = {list(range(answer_seq_len - 1))}")
    for t in range(answer_seq_len - 1):
        input_tok = answer[0] if t == 0 else answer[t]
        label_tok = answer[t + 1]
        print(f"  t={t}: input={input_tok.item()}, outputs[{t}] -> label={label_tok.item()}")
    print(f"Labels: answer[1:] = {answer[1:].squeeze().tolist()}")
    print(f"Predicts: [A, B, EOS] ✓")
    
    print(f"\n--- Proposed Fix ---")
    print(f"Loop range: range({answer_seq_len - 2}) = {list(range(answer_seq_len - 2))}")
    for t in range(answer_seq_len - 2):
        input_tok = answer[0] if t == 0 else answer[t]
        label_tok = answer[t + 1]
        print(f"  t={t}: input={input_tok.item()}, outputs[{t}] -> label={label_tok.item()}")
    print(f"Labels: answer[1:-1] = {answer[1:-1].squeeze().tolist()}")
    print(f"Predicts: [A, B] ❌ Missing EOS!")
    
    print(f"\n⚠️  PROBLEM: The fix breaks unpadded sequences!")
    print(f"    We need a different approach...")


if __name__ == "__main__":
    bug_exists = test_current_implementation()
    fix_works_padded = test_proposed_fix()
    test_unpadded_sequence()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print(f"Bug exists in current implementation: {bug_exists}")
    print(f"Proposed fix works for padded sequences: {fix_works_padded}")
    print(f"But proposed fix breaks unpadded sequences!")
    print(f"\nWe need to handle both padded and unpadded sequences correctly.")
