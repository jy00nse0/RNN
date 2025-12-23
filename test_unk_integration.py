#!/usr/bin/env python3
"""
Integration test for UNK replacement - validates the complete workflow.
This test doesn't require trained models, just checks the interfaces.
"""

import numpy as np


def test_sampling_signature():
    """Test that sampling methods have the correct signature."""
    # We just check that the file compiles and has the right structure
    import inspect
    from model.seq2seq.sampling import GreedySampler, RandomSampler, BeamSearch
    
    # Check GreedySampler.sample signature
    greedy = GreedySampler()
    sig = inspect.signature(greedy.sample)
    params = list(sig.parameters.keys())
    assert 'return_attention' in params, "GreedySampler.sample should have return_attention parameter"
    print("✓ GreedySampler has return_attention parameter")
    
    # Check RandomSampler.sample signature
    random = RandomSampler()
    sig = inspect.signature(random.sample)
    params = list(sig.parameters.keys())
    assert 'return_attention' in params, "RandomSampler.sample should have return_attention parameter"
    print("✓ RandomSampler has return_attention parameter")
    
    # Check BeamSearch.sample signature
    beam = BeamSearch()
    sig = inspect.signature(beam.sample)
    params = list(sig.parameters.keys())
    assert 'return_attention' in params, "BeamSearch.sample should have return_attention parameter"
    print("✓ BeamSearch has return_attention parameter")


def test_model_signature():
    """Test that Seq2SeqPredict.forward has the correct signature."""
    import inspect
    from model.seq2seq.model import Seq2SeqPredict
    
    sig = inspect.signature(Seq2SeqPredict.forward)
    params = list(sig.parameters.keys())
    assert 'return_attention' in params, "Seq2SeqPredict.forward should have return_attention parameter"
    print("✓ Seq2SeqPredict.forward has return_attention parameter")


def test_calculate_bleu_imports():
    """Test that calculate_bleu has the required functions."""
    from calculate_bleu import unk_replace, parse_args
    
    # Check unk_replace is callable
    assert callable(unk_replace), "unk_replace should be callable"
    print("✓ unk_replace function exists and is callable")
    
    # Check parse_args includes unk-replace argument
    parser = parse_args.__code__.co_consts
    # Just check the function exists for now
    assert callable(parse_args), "parse_args should be callable"
    print("✓ parse_args function exists")


def test_unk_replace_with_none_attention():
    """Test that unk_replace handles None attention gracefully."""
    from calculate_bleu import unk_replace
    
    hypothesis = "The <unk> is very important ."
    source = "Das Buch ist sehr wichtig ."
    
    result = unk_replace(hypothesis, source, None)
    assert result == hypothesis, "Should return unchanged when attention is None"
    print("✓ unk_replace handles None attention correctly")


def test_unk_replace_preserves_non_unk():
    """Test that unk_replace doesn't modify sentences without UNK."""
    from calculate_bleu import unk_replace
    
    hypothesis = "The book is very important ."
    source = "Das Buch ist sehr wichtig ."
    attention = np.zeros((6, 6))
    
    result = unk_replace(hypothesis, source, attention)
    assert result == hypothesis, "Should preserve sentences without UNK"
    print("✓ unk_replace preserves non-UNK sentences")


def run_integration_tests():
    """Run all integration tests."""
    print("="*70)
    print("Running UNK Replacement Integration Tests")
    print("="*70 + "\n")
    
    test_sampling_signature()
    test_model_signature()
    test_calculate_bleu_imports()
    test_unk_replace_with_none_attention()
    test_unk_replace_preserves_non_unk()
    
    print("\n" + "="*70)
    print("All integration tests PASSED! ✓")
    print("="*70)


if __name__ == '__main__':
    run_integration_tests()
