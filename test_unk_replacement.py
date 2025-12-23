#!/usr/bin/env python3
"""
Unit tests for UNK replacement functionality.

Tests the UNK replacement technique from Luong et al. (2015)
"Effective Approaches to Attention-based Neural Machine Translation".
"""

import numpy as np


def unk_replace(hypothesis, source, attention_weights, unk_token='<unk>', dictionary=None):
    """
    Replace <unk> tokens in hypothesis with source words using attention weights.
    
    Implementation of the UNK replacement technique from Luong et al. (2015)
    "Effective Approaches to Attention-based Neural Machine Translation".
    
    Args:
        hypothesis (str): Generated translation containing <unk> tokens
        source (str): Source sentence (tokenized string)
        attention_weights (numpy array or None): Attention weights of shape (tgt_len, src_len)
        unk_token (str): The unknown token string (default: '<unk>')
        dictionary (dict or None): Optional bilingual dictionary for word translation
        
    Returns:
        str: Hypothesis with <unk> tokens replaced
    """
    # If no attention weights available, cannot perform replacement
    if attention_weights is None:
        return hypothesis
    
    # Split hypothesis and source into tokens
    hyp_tokens = hypothesis.split()
    src_tokens = source.split()
    
    # If no UNK tokens, return as-is
    if unk_token not in hyp_tokens:
        return hypothesis
    
    # Convert to numpy for easier manipulation if needed
    if hasattr(attention_weights, 'cpu'):
        # torch tensor
        attention_weights = attention_weights.cpu().numpy()
    
    # Process each token in hypothesis
    replaced_tokens = []
    for t, token in enumerate(hyp_tokens):
        if token == unk_token:
            # Find source position with highest attention
            if t < attention_weights.shape[0]:
                # Get attention distribution for this target position
                attn_dist = attention_weights[t]
                # Find source position with max attention
                src_pos = attn_dist.argmax()
                
                # Get the source word
                if src_pos < len(src_tokens):
                    src_word = src_tokens[src_pos]
                    
                    # If dictionary provided, try to translate
                    if dictionary is not None and src_word in dictionary:
                        replaced_tokens.append(dictionary[src_word])
                    else:
                        # Direct copy (for proper nouns, numbers, etc.)
                        replaced_tokens.append(src_word)
                else:
                    # Fallback: keep UNK
                    replaced_tokens.append(token)
            else:
                # Fallback: keep UNK
                replaced_tokens.append(token)
        else:
            replaced_tokens.append(token)
    
    return ' '.join(replaced_tokens)


def test_unk_replace_basic():
    """Test basic UNK replacement with direct copy."""
    hypothesis = "The <unk> is very important ."
    source = "Das Buch ist sehr wichtig ."
    
    # Create attention weights where position 1 (Buch) has highest attention for <unk>
    # Target tokens: The <unk> is very important .
    # Source tokens: Das Buch ist sehr wichtig .
    attention_weights = np.zeros((6, 7))  # (tgt_len, src_len)
    attention_weights[1, 1] = 1.0  # <unk> at position 1 attends to "Buch" at position 1
    
    result = unk_replace(hypothesis, source, attention_weights)
    expected = "The Buch is very important ."
    
    print(f"Input:    {hypothesis}")
    print(f"Source:   {source}")
    print(f"Expected: {expected}")
    print(f"Result:   {result}")
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print("✓ test_unk_replace_basic PASSED\n")


def test_unk_replace_multiple():
    """Test UNK replacement with multiple UNK tokens."""
    hypothesis = "The <unk> contains <unk> information ."
    source = "Das Buch enthält wichtige Informationen ."
    
    # Target: The <unk> contains <unk> information .
    # Source: Das Buch enthält wichtige Informationen .
    attention_weights = np.zeros((6, 6))
    attention_weights[1, 1] = 1.0  # First <unk> -> Buch
    attention_weights[3, 3] = 1.0  # Second <unk> -> wichtige
    
    result = unk_replace(hypothesis, source, attention_weights)
    expected = "The Buch contains wichtige information ."
    
    print(f"Input:    {hypothesis}")
    print(f"Source:   {source}")
    print(f"Expected: {expected}")
    print(f"Result:   {result}")
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print("✓ test_unk_replace_multiple PASSED\n")


def test_unk_replace_no_unk():
    """Test that sentences without UNK are unchanged."""
    hypothesis = "The book is very important ."
    source = "Das Buch ist sehr wichtig ."
    attention_weights = np.zeros((6, 6))
    
    result = unk_replace(hypothesis, source, attention_weights)
    
    print(f"Input:    {hypothesis}")
    print(f"Result:   {result}")
    assert result == hypothesis, f"Expected unchanged hypothesis, got '{result}'"
    print("✓ test_unk_replace_no_unk PASSED\n")


def test_unk_replace_no_attention():
    """Test that UNK is preserved when no attention weights available."""
    hypothesis = "The <unk> is very important ."
    source = "Das Buch ist sehr wichtig ."
    attention_weights = None
    
    result = unk_replace(hypothesis, source, attention_weights)
    
    print(f"Input:    {hypothesis}")
    print(f"Result:   {result}")
    assert result == hypothesis, f"Expected unchanged hypothesis, got '{result}'"
    print("✓ test_unk_replace_no_attention PASSED\n")


def test_unk_replace_with_dictionary():
    """Test UNK replacement with bilingual dictionary."""
    hypothesis = "The <unk> is very important ."
    source = "Das Buch ist sehr wichtig ."
    
    # Bilingual dictionary
    dictionary = {
        'Buch': 'book',
        'sehr': 'very',
        'wichtig': 'important'
    }
    
    attention_weights = np.zeros((6, 7))
    attention_weights[1, 1] = 1.0  # <unk> -> Buch
    
    result = unk_replace(hypothesis, source, attention_weights, dictionary=dictionary)
    expected = "The book is very important ."
    
    print(f"Input:    {hypothesis}")
    print(f"Source:   {source}")
    print(f"Expected: {expected}")
    print(f"Result:   {result}")
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print("✓ test_unk_replace_with_dictionary PASSED\n")


def test_unk_replace_soft_attention():
    """Test with realistic soft attention distribution."""
    hypothesis = "The <unk> is important ."
    source = "Das Buch ist wichtig ."
    
    # Realistic soft attention (not all weight on one position)
    attention_weights = np.zeros((5, 5))
    # Second word (<unk>) attends mostly to "Buch" but with some spread
    attention_weights[1] = np.array([0.1, 0.6, 0.2, 0.1, 0.0])
    
    result = unk_replace(hypothesis, source, attention_weights)
    expected = "The Buch is important ."  # Should pick argmax (position 1)
    
    print(f"Input:    {hypothesis}")
    print(f"Source:   {source}")
    print(f"Expected: {expected}")
    print(f"Result:   {result}")
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print("✓ test_unk_replace_soft_attention PASSED\n")


def run_all_tests():
    """Run all test functions."""
    print("="*70)
    print("Running UNK Replacement Tests")
    print("="*70 + "\n")
    
    test_unk_replace_basic()
    test_unk_replace_multiple()
    test_unk_replace_no_unk()
    test_unk_replace_no_attention()
    test_unk_replace_with_dictionary()
    test_unk_replace_soft_attention()
    
    print("="*70)
    print("All tests PASSED! ✓")
    print("="*70)


if __name__ == '__main__':
    run_all_tests()
