# UNK Replacement Feature

## Overview
This implementation adds the **UNK Replacement** technique described in Luong et al. (2015) "Effective Approaches to Attention-based Neural Machine Translation". This post-processing technique uses attention weights to replace `<unk>` tokens in generated translations with appropriate source words.

## How It Works

1. When the decoder generates `<unk>` at position `t`, the technique identifies the source word with the highest attention: `i = argmax_s α_{t,s}`
2. The `<unk>` token is replaced with:
   - Translation from a bilingual dictionary (if available)
   - Direct copy of the source word `x_i` (for proper nouns, numbers, etc.)

According to the original paper, this simple post-processing improved BLEU scores by +1.9 points.

## Usage

### Basic Usage (Direct Copy)

To use UNK replacement with direct copying of source words:

```bash
python calculate_bleu.py \
    --model-path checkpoints/my_model/ \
    --reference-path data/test.de \
    --epoch 10 \
    --unk-replace \
    --cuda
```

The `--unk-replace` flag enables the UNK replacement feature.

### With Bilingual Dictionary

To use UNK replacement with a bilingual dictionary (future enhancement):

```python
# In calculate_bleu.py, you can modify the unk_replace call to include a dictionary
dictionary = {
    'Buch': 'book',
    'schnell': 'fast',
    # ... more translations
}

replaced_answer = unk_replace(answer, question, attn, dictionary=dictionary)
```

## Implementation Details

### Modified Files

1. **model/seq2seq/sampling.py**
   - `GreedySampler.sample()`: Added `return_attention` parameter to optionally return attention weights
   - `RandomSampler.sample()`: Same as above
   - `BeamSearch.sample()`: Same as above (returns None for attention weights)

2. **model/seq2seq/model.py**
   - `Seq2SeqPredict.forward()`: Added `return_attention` parameter to propagate attention weights to callers

3. **calculate_bleu.py**
   - `unk_replace()`: New function that implements the UNK replacement logic
   - `parse_args()`: Added `--unk-replace` CLI argument
   - `get_answers()`: Modified to support returning attention weights
   - `main()`: Modified to apply UNK replacement when requested

### Function Signature

```python
def unk_replace(hypothesis, source, attention_weights, unk_token='<unk>', dictionary=None):
    """
    Replace <unk> tokens in hypothesis with source words using attention weights.
    
    Args:
        hypothesis (str): Generated translation containing <unk> tokens
        source (str): Source sentence (tokenized string)
        attention_weights (torch.Tensor or None): Attention weights of shape (tgt_len, src_len)
        unk_token (str): The unknown token string (default: '<unk>')
        dictionary (dict or None): Optional bilingual dictionary for word translation
        
    Returns:
        str: Hypothesis with <unk> tokens replaced
    """
```

## Testing

Run the unit tests:

```bash
python test_unk_replacement.py
```

This will test:
- Basic UNK replacement with direct copy
- Multiple UNK tokens in one sentence
- Sentences without UNK tokens (should remain unchanged)
- Handling of missing attention weights
- Bilingual dictionary translation
- Soft attention distributions
- Numpy array compatibility

## Limitations

1. **BeamSearch**: Currently, BeamSearch sampling does not track attention weights, so it returns `None`. UNK replacement will not be applied when using beam search.

2. **Models without attention**: If a model doesn't use attention mechanisms (e.g., `attention_type='none'`), the attention weights will be `None` and UNK replacement will not be applied.

3. **Performance**: UNK replacement is a post-processing step and requires attention weights to be stored in memory, which may increase memory usage for large batches.

## Examples

### Example 1: Direct Copy
```
Source:      "Das Buch ist sehr wichtig ."
Hypothesis:  "The <unk> is very important ."
Attention:   Position 1 has max attention at source position 1 (Buch)
Result:      "The Buch is very important ."
```

### Example 2: With Dictionary
```
Source:      "Das Buch ist sehr wichtig ."
Hypothesis:  "The <unk> is very important ."
Dictionary:  {'Buch': 'book'}
Attention:   Position 1 has max attention at source position 1 (Buch)
Result:      "The book is very important ."
```

### Example 3: Multiple UNK Tokens
```
Source:      "Das Buch enthält wichtige Informationen ."
Hypothesis:  "The <unk> contains <unk> information ."
Attention:   Position 1 → Buch, Position 3 → wichtige
Result:      "The Buch contains wichtige information ."
```

## Future Enhancements

1. **Bilingual Dictionary Loading**: Add support for loading bilingual dictionaries from files
2. **BeamSearch Support**: Track attention weights during beam search decoding
3. **Configurable UNK Token**: Allow customization of the UNK token string
4. **Batch Processing**: Optimize for processing large batches of sentences

## References

Luong, M. T., Pham, H., & Manning, C. D. (2015). Effective Approaches to Attention-based Neural Machine Translation. arXiv preprint arXiv:1508.04025.
