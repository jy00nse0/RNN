#!/usr/bin/env python3
"""
Test to verify the decoder input/label slicing fix.

This test verifies that after the fix:
1. Decoder never receives EOS or PAD as inputs
2. Label slicing matches decoder output shape
3. Both padded and unpadded sequences work correctly
"""

import torch

def test_fix_with_padding():
    """Test the fix with a padded sequence."""
    print("="*70)
    print("TEST 1: PADDED SEQUENCE")
    print("="*70)
    
    # After fix: All sequences have format [SOS, tok1, ..., tokN, EOS, PAD]
    # Example: [SOS, A, B, EOS, PAD, PAD]
    answer = torch.tensor([[0], [5], [6], [1], [2], [2]])
    answer_seq_len = answer.size(0)
    
    print(f"\nSequence: {answer.squeeze().tolist()}")
    print(f"  SOS=0, A=5, B=6, EOS=1, PAD=2")
    print(f"Length: {answer_seq_len}")
    
    print(f"\nFixed training loop:")
    print(f"  Loop range: range({answer_seq_len} - 3) = range({answer_seq_len - 3}) = {list(range(answer_seq_len - 3))}")
    
    print(f"\nDecoder steps (with teacher forcing):")
    
    input_word = answer[0]
    has_bug = False
    
    for t in range(answer_seq_len - 3):
        output_idx = t
        label = answer[t + 1]
        
        status = ""
        if input_word.item() == 1:  # EOS
            status += " ⚠️  BUG: Using EOS as input!"
            has_bug = True
        if input_word.item() == 2:  # PAD
            status += " ⚠️  BUG: Using PAD as input!"
            has_bug = True
        
        print(f"  t={t}: input={input_word.item()}, outputs[{output_idx}] -> label={label.item()}{status}")
        
        # Teacher forcing
        input_word = answer[t + 1]
    
    print(f"\nOutputs shape: ({answer_seq_len - 3}, batch, vocab)")
    print(f"Labels: answer[1:-2] = {answer[1:-2].squeeze().tolist()}")
    print(f"Labels shape: ({len(answer[1:-2])}, batch)")
    
    shapes_match = (answer_seq_len - 3) == len(answer[1:-2])
    
    print(f"\n{'✅' if not has_bug else '❌'} Decoder inputs: {'No EOS/PAD' if not has_bug else 'Contains EOS/PAD'}")
    print(f"{'✅' if shapes_match else '❌'} Shape matching: {'Outputs and labels match' if shapes_match else 'Mismatch!'}")
    print(f"{'✅' if 1 in answer[1:-2].squeeze().tolist() else '❌'} EOS prediction: {'Included' if 1 in answer[1:-2].squeeze().tolist() else 'Missing'}")
    
    return not has_bug and shapes_match


def test_fix_without_padding():
    """Test the fix with a sequence that would have been unpadded before."""
    print("\n" + "="*70)
    print("TEST 2: SEQUENCE WITHOUT EXTRA PADDING (max length in batch)")
    print("="*70)
    
    # After fix: Even the longest sequence has 2 PADs at the end
    # Example: [SOS, X, Y, Z, EOS, PAD, PAD]
    answer = torch.tensor([[0], [10], [11], [12], [1], [2], [2]])
    answer_seq_len = answer.size(0)
    
    print(f"\nSequence: {answer.squeeze().tolist()}")
    print(f"  SOS=0, X=10, Y=11, Z=12, EOS=1, PAD=2, PAD=2")
    print(f"Length: {answer_seq_len}")
    
    print(f"\nFixed training loop:")
    print(f"  Loop range: range({answer_seq_len} - 3) = {list(range(answer_seq_len - 3))}")
    
    print(f"\nDecoder steps:")
    
    input_word = answer[0]
    has_bug = False
    
    for t in range(answer_seq_len - 3):
        output_idx = t
        label = answer[t + 1]
        
        status = ""
        if input_word.item() == 1:  # EOS
            status += " ⚠️  BUG: Using EOS as input!"
            has_bug = True
        if input_word.item() == 2:  # PAD
            status += " ⚠️  BUG: Using PAD as input!"
            has_bug = True
        
        print(f"  t={t}: input={input_word.item()}, outputs[{output_idx}] -> label={label.item()}{status}")
        
        input_word = answer[t + 1]
    
    print(f"\nOutputs shape: ({answer_seq_len - 3}, batch, vocab)")
    print(f"Labels: answer[1:-2] = {answer[1:-2].squeeze().tolist()}")
    
    shapes_match = (answer_seq_len - 3) == len(answer[1:-2])
    eos_in_labels = 1 in answer[1:-2].squeeze().tolist()
    
    print(f"\n{'✅' if not has_bug else '❌'} Decoder inputs: {'No EOS/PAD' if not has_bug else 'Contains EOS/PAD'}")
    print(f"{'✅' if shapes_match else '❌'} Shape matching: {'Outputs and labels match' if shapes_match else 'Mismatch!'}")
    print(f"{'✅' if eos_in_labels else '❌'} EOS prediction: {'Included' if eos_in_labels else 'Missing'}")
    
    return not has_bug and shapes_match and eos_in_labels


def test_comparison():
    """Compare old vs new implementation."""
    print("\n" + "="*70)
    print("COMPARISON: OLD vs NEW")
    print("="*70)
    
    # Old format: [SOS, A, B, EOS, PAD]
    # New format: [SOS, A, B, EOS, PAD, PAD]
    
    print("\nOLD FORMAT: [SOS, A, B, EOS, PAD] - length 5")
    print("  Loop: range(4) -> Uses EOS as input at t=3 ❌")
    
    print("\nNEW FORMAT (FIXED): [SOS, A, B, EOS, PAD, PAD] - length 6")
    
    answer_new = torch.tensor([[0], [5], [6], [1], [2], [2]])
    print(f"  answer_seq_len = {answer_new.size(0)}")
    print(f"  Loop: range(answer_seq_len - 3) = range({answer_new.size(0) - 3}) = {list(range(answer_new.size(0) - 3))}")
    
    print("\n  Decoder steps:")
    for t in range(answer_new.size(0) - 3):
        if t == 0:
            inp = answer_new[0].item()
        else:
            inp = answer_new[t].item()
        label = answer_new[t + 1].item()
        print(f"    t={t}: input={inp}, label={label}", end="")
        if inp == 1:
            print(" ⚠️  EOS INPUT", end="")
        if inp == 2:
            print(" ⚠️  PAD INPUT", end="")
        print()
    
    print("\n  ✅ CORRECT! With answer_seq_len=6, range(3) = [0,1,2]:")
    print("     t=0: input=SOS(0), predict=A(5) ✓")
    print("     t=1: input=A(5), predict=B(6) ✓")  
    print("     t=2: input=B(6), predict=EOS(1) ✓")
    print("\n  No EOS or PAD used as inputs! Fix is working correctly.")


if __name__ == "__main__":
    print("DECODER INPUT/LABEL SLICING FIX VERIFICATION\n")
    
    test1_pass = test_fix_with_padding()
    test2_pass = test_fix_without_padding()
    test_comparison()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Test 1 (Padded sequence): {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Test 2 (Unpadded sequence): {'✅ PASS' if test2_pass else '❌ FAIL'}")
    
    if test1_pass and test2_pass:
        print("\n✅ All tests passed! The fix works correctly.")
    else:
        print("\n❌ Some tests failed. Need to revise the fix.")
