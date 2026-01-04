# Fix for BLEU Evaluation Vocab Mismatch Issue

## Summary
This PR fixes a critical bug in BLEU evaluation where embedding size mismatch errors occur when source and target vocabularies have different sizes.

## Problem Description
During BLEU evaluation using `calculate_bleu.py`, the following error occurred:
```
RuntimeError: Error(s) in loading state_dict for Seq2SeqTrain:
    size mismatch for encoder.embed.weight: copying a param with shape torch.Size([7807, 1000]) 
    from checkpoint, the shape in current model is torch.Size([10150, 1000]).
```

### Root Cause
1. During training, only `tgt_vocab` (target vocabulary) was saved
2. During evaluation, `calculate_bleu.py` loaded this single vocabulary and used it for both source and target
3. When source and target vocabularies have different sizes, this caused a mismatch between:
   - Checkpoint's encoder embedding size (source vocab size during training)
   - Evaluation model's encoder embedding size (target vocab size, incorrectly reused)

## Solution

### Changes to `train.py`
```python
# Before: Only saved tgt_vocab
save_vocab(tgt_vocab, os.path.join(args.save_path, 'vocab'))

# After: Save both vocabularies
save_vocab(src_vocab, os.path.join(args.save_path, 'src_vocab'))
save_vocab(tgt_vocab, os.path.join(args.save_path, 'tgt_vocab'))
save_vocab(tgt_vocab, os.path.join(args.save_path, 'vocab'))  # Backward compatibility
```

### Changes to `calculate_bleu.py`
1. **Load separate vocabularies**: Load `src_vocab` and `tgt_vocab` separately when available
2. **Backward compatibility**: Fall back to single `vocab` file for old checkpoints
3. **Size validation**: For old checkpoints, validate that vocab sizes match checkpoint embedding sizes
4. **Error handling**: Provide clear error message when src_vocab is missing and sizes don't match

```python
# New format: Load separate vocabularies
if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
    src_vocab = load_object(src_vocab_path)
    tgt_vocab = load_object(tgt_vocab_path)
    print(f"Loaded separate vocabularies: src_vocab ({len(src_vocab)}), tgt_vocab ({len(tgt_vocab)})")
    
# Old format: Validate sizes before using single vocab for both
elif os.path.exists(vocab_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    encoder_embed_size = checkpoint['encoder.embed.weight'].shape[0]
    decoder_embed_size = checkpoint['decoder.embed.weight'].shape[0]
    tgt_vocab = load_object(vocab_path)
    
    if encoder_embed_size == decoder_embed_size == len(tgt_vocab):
        src_vocab = tgt_vocab  # Safe to reuse
    else:
        # Clear error message
        raise RuntimeError("Cannot load model: src_vocab file missing and vocab sizes don't match")
```

## Testing

### Unit Tests (`test_vocab_saving.py`)
- ✅ Test vocabulary saving and loading
- ✅ Test checkpoint embedding size extraction

### Integration Tests (`test_vocab_integration.py`)
- ✅ Test training saves both vocabularies
- ✅ Test evaluation loads both vocabularies correctly
- ✅ Test model can be loaded with separate vocabularies

### Manual Verification (`manual_verification.py`)
- ✅ End-to-end test with real training
- ✅ BLEU evaluation with separate vocabularies
- ✅ No embedding size mismatch errors
- ✅ Verified checkpoint embedding sizes match vocab sizes

All tests pass: 4/4 unit/integration tests + manual verification ✅

## Backward Compatibility
- Old checkpoints with only `vocab` file: Supported if source and target vocab sizes match
- New checkpoints: Always save both `src_vocab` and `tgt_vocab`
- Migration path: Re-run training with updated `train.py` to generate new checkpoints with both vocabularies

## Files Changed
1. `train.py`: Save both src_vocab and tgt_vocab
2. `calculate_bleu.py`: Load and validate separate vocabularies
3. `dataset.py`: Minor fix for spacing and add manual_test dataset support
4. `test_vocab_saving.py`: Unit tests (new)
5. `test_vocab_integration.py`: Integration tests (new)
6. `manual_verification.py`: Manual end-to-end test (new)

## Impact
- **Breaking**: None (backward compatible)
- **Performance**: No impact
- **Security**: No security issues (verified with CodeQL)
- **User Action Required**: For old checkpoints with mismatched vocab sizes, re-train the model

## Verification Steps
To verify the fix works:

1. Run unit tests:
   ```bash
   python test_vocab_saving.py
   ```

2. Run integration tests:
   ```bash
   python test_vocab_integration.py
   ```

3. Run manual verification:
   ```bash
   python manual_verification.py
   ```

4. Or test with actual sample data:
   ```bash
   python sample_test.py
   ```

All tests should pass without embedding size mismatch errors.
