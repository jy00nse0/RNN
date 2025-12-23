# Perplexity (PPL) Calculation Implementation

## Overview
This implementation adds Perplexity (PPL) calculation to the RNN training and evaluation pipeline, as requested in the issue.

## What is Perplexity?
Perplexity is a measurement of how well a probability model predicts a sample. For language models:
- PPL = exp(Cross Entropy Loss)
- Lower PPL indicates better model performance
- PPL represents the average branching factor of the model's predictions

## Implementation Details

### 1. New File: `calculate_ppl.py`
A standalone script for calculating PPL, similar to `calculate_bleu.py`.

**Key Features:**
- Loads a trained model checkpoint from a specific epoch
- Evaluates the model on the test dataset
- Computes cross-entropy loss (ignoring padding tokens)
- Calculates PPL = exp(loss)
- Outputs results in a parseable format

**Usage:**
```bash
python calculate_ppl.py --model-path <path> --epoch <num> --dataset <dataset_name> [--cuda]
```

**Example:**
```bash
python calculate_ppl.py --model-path checkpoints/T1_Base_Reverse --epoch 10 --dataset wmt14-en-de --cuda
```

**Output:**
```
Loading dataset: wmt14-en-de
Loading model from: checkpoints/T1_Base_Reverse/seq2seq-10-2.5431-2.8764.pt
Evaluating model on test set...

Test Loss: 2.8764
Perplexity (PPL): 17.7542
PPL = 17.7542
```

### 2. Modified File: `test_new.py`

#### Added Function: `evaluate_ppl()`
```python
def evaluate_ppl(save_path, dataset_name, epoch, common_flags):
    """
    Calculate Perplexity (PPL) for a specific epoch by evaluating test loss.
    
    Args:
        save_path: Path to model checkpoint directory
        dataset_name: Dataset name (e.g., 'wmt14-en-de')
        epoch: Epoch number to evaluate
        common_flags: Common flags (e.g., --cuda)
    
    Returns:
        PPL as float, or None if evaluation failed
    """
```

This function:
1. Constructs command to run `calculate_ppl.py`
2. Executes the command and captures output
3. Parses PPL value from the output
4. Returns the PPL score or None if evaluation fails

#### Modified Training Loop
The training loop now:
1. Calculates BLEU score for every epoch (unchanged)
2. **NEW:** Calculates PPL for the final epoch only
3. Logs both BLEU and PPL to the log file for the final epoch

**Example log output:**
```
Epoch  1:  BLEU = 15.23
Epoch  2:  BLEU = 18.45
...
Epoch 10:  BLEU = 25.34, PPL = 45.67
```

#### Handling Already-Completed Training
If training has already been completed, the script now:
1. Detects the completion
2. Evaluates PPL for the final epoch
3. Appends the result to the log file

## Why PPL is Calculated Only for the Last Epoch

Per the requirement: "test_new.py에서 맨 마지막 에포크 bleu 계산 시 ppl 도 계산 및 출력되도록 할것"

This design choice:
1. **Reduces computational overhead** - PPL calculation requires loading the model and evaluating on the test set
2. **Focuses on final model quality** - PPL for intermediate epochs is less relevant
3. **Matches the requirement** - Only the last epoch needs PPL alongside BLEU

## Mathematical Background

The relationship between loss and perplexity:

```
Cross Entropy Loss = -(1/N) * Σ log P(y_i)
PPL = exp(Cross Entropy Loss)
```

Where:
- N = number of tokens (excluding padding)
- P(y_i) = probability assigned to the correct token i
- Lower loss → Lower PPL → Better model

## Integration with Existing Code

The implementation reuses existing infrastructure:
- Uses the same model loading mechanism as `calculate_bleu.py`
- Uses the same `dataset_factory` and `metadata_factory` functions
- Uses the same evaluation loop structure as `train.py`
- Follows the same command-line interface patterns

## Testing

To test the implementation:

1. **Syntax check (no dependencies required):**
```bash
python -m py_compile calculate_ppl.py
python -m py_compile test_new.py
```

2. **With a trained model:**
```bash
python calculate_ppl.py --model-path <checkpoint_dir> --epoch 10 --dataset wmt14-en-de --cuda
```

3. **Full integration test:**
```bash
python test_new.py -e T1_Base_Reverse
# This will calculate both BLEU and PPL for the final epoch
```

## Files Modified

1. **calculate_ppl.py** (NEW) - Standalone PPL calculation script
2. **test_new.py** (MODIFIED) - Added PPL calculation to training loop
   - Added imports: `math`, `torch.nn.functional as F`, `DataLoader`, `tqdm`
   - Added `evaluate_ppl()` function
   - Modified training loop to calculate PPL for final epoch
   - Added PPL evaluation for already-completed runs
   - Fixed pre-existing syntax error: `1. 0` → `1.0`

## Dependencies

No new dependencies were added. The implementation uses:
- `torch` - Already required
- `math` - Python standard library
- `tqdm` - Already required
- Other existing modules from the codebase

## Future Enhancements

Possible improvements (not implemented in this PR):
1. Calculate PPL for all epochs (if computational resources allow)
2. Add PPL tracking during training (in `train.py`)
3. Create visualization of PPL vs BLEU correlation
4. Add early stopping based on PPL threshold
