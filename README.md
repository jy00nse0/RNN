# RNN-based Neural Machine Translation (WMT14 En-De)

This repository implements an RNN-based sequence-to-sequence model for English-to-German translation using the WMT14 dataset.

## Features

- Encoder-Decoder architecture with attention mechanism
- BPE (Byte Pair Encoding) tokenization using SentencePiece
- Beam search decoding
- BLEU score evaluation using sacreBLEU

## Setup

### Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch (>=1.9.0)
- Hugging Face datasets (>=2.0.0)
- SentencePiece (>=0.1.96)
- sacreBLEU (>=2.0.0)

## Data Preparation

### Step 1: Download and Preprocess Data

The new data download script uses Hugging Face datasets to download WMT14 En-De data:

```bash
python scripts/download_data.py
```

This will:
- Download WMT14 English-German parallel corpus from Hugging Face
- Apply preprocessing: filter sentences with length ≤50 words
- Reverse source (English) sentences for better RNN performance
- Save preprocessed data to `data/wmt14_raw/`:
  - `train.clean.en` / `train.clean.de`
  - `valid.clean.en` / `valid.clean.de`
  - `test.clean.en` / `test.clean.de`

### Step 2: Build BPE Vocabulary

Train a BPE model and apply it to the data:

```bash
python scripts/build_bpe.py
```

This will:
- Train a SentencePiece BPE model on the combined En+De training data
- Apply BPE tokenization to train/valid/test splits
- Save BPE-encoded data to `data/wmt14_bpe/`:
  - `train.bpe.en` / `train.bpe.de`
  - `valid.bpe.en` / `valid.bpe.de`
  - `test.bpe.en` / `test.bpe.de`
  - `bpe.model` / `bpe.vocab`

## Training

Train the model:

```bash
python train/main.py --data_dir data/wmt14_bpe --epochs 10 --batch_size 128
```

Arguments:
- `--data_dir`: Directory containing BPE-encoded data
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--lr`: Learning rate (default: 1.0)
- `--lr_decay_start`: Epoch to start learning rate decay (default: 5)
- `--beam_size`: Beam size for validation BLEU calculation (default: 10)
- `--save_dir`: Directory to save checkpoints (default: checkpoints/)

The script will:
- Train the model on the training set
- Evaluate BLEU score on the validation set after each epoch
- Save the best checkpoint based on validation BLEU

## Testing

Evaluate the trained model on the test set:

```bash
python test/test.py --data_dir data/wmt14_bpe --ckpt_path checkpoints/best_epoch5.pt
```

Arguments:
- `--data_dir`: Directory containing BPE-encoded data
- `--ckpt_path`: Path to the saved model checkpoint
- `--beam_size`: Beam size for beam search (default: 10)
- `--max_len`: Maximum generation length (default: 100)

## Model Architecture

- **Encoder**: Multi-layer bidirectional LSTM
- **Decoder**: Multi-layer LSTM with attention mechanism
- **Embedding dimension**: 1000
- **Hidden dimension**: 1000
- **Number of layers**: 4

## Data Preprocessing Details

The preprocessing pipeline follows the original neural machine translation paper:

1. **Length filtering**: Keep only sentence pairs where both source and target have ≤50 words
2. **Source reversing**: Reverse the order of words in English sentences to improve RNN training
3. **BPE tokenization**: Apply Byte Pair Encoding with vocabulary size 50,000

## Notes

- The source (English) sentences are reversed during preprocessing to help the RNN learn better alignments
- Data is NOT reversed again during training/inference (reversing is done once in `download_data.py`)
- The validation split corresponds to newstest2013
- The test split corresponds to newstest2014