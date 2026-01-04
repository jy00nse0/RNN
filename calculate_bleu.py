#!/usr/bin/env python

import torch
import pandas as pd
import os
import argparse
import sacrebleu
from model import predict_model_factory
from dataset import field_factory, metadata_factory
from serialization import load_object
from constants import MODEL_START_FORMAT


def parse_args():
    parser = argparse.ArgumentParser(description='Script for calculating chatbot BLEU score.')
    parser.add_argument('-p', '--model-path', required=True,
                        help='Path to directory with model args, vocabulary and pre-trained pytorch models.')
    parser.add_argument('-e', '--epoch', type=int, help='Model from this epoch will be loaded.')
    parser.add_argument('--sampling-strategy', choices=['greedy', 'random', 'beam_search'], default='greedy',
                        help='Strategy for sampling output sequence.')
    parser.add_argument('-r', '--reference-path', required=True, help='Path to reference file.')
    parser.add_argument('--max-seq-len', type=int, default=30, help='Maximum length for output sequence.')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use cuda if available.')
    return parser.parse_args()


def get_model_path(dir_path, epoch):
    name_start = MODEL_START_FORMAT % epoch
    for path in os.listdir(dir_path):
        if path.startswith(name_start):
            return os.path.join(dir_path, path)
    raise ValueError("Model from epoch %d doesn't exist in %s" % (epoch, dir_path))


def get_answers(model, questions, args):
    batch_size = 1000
    answers = []
    num_batches = len(questions) // batch_size
    rest = len(questions) % batch_size
    for batch in range(num_batches):
        batch_answers = model(questions[batch * batch_size:(batch + 1) * batch_size],
                             sampling_strategy=args.sampling_strategy,
                             max_seq_len=args.max_seq_len)
        answers.extend(batch_answers)

    if rest != 0:
        batch_answers = model(questions[-rest:], sampling_strategy=args.sampling_strategy,
                             max_seq_len=args.max_seq_len)
        answers.extend(batch_answers)

    return answers


class SimpleField:
    """Simple field-like object that wraps vocab for compatibility"""
    def __init__(self, vocab):
        self.vocab = vocab
    
    def preprocess(self, text):
        """Tokenize text by splitting on whitespace"""
        return text.strip().split()
    
    def process(self, batch):
        """Convert list of token lists to tensor"""
        max_len = max(len(tokens) for tokens in batch)
        # Add <sos> and <eos> tokens
        processed = []
        for tokens in batch:
            indices = [self.vocab.stoi['<sos>']] + \
                      [self.vocab.stoi.get(tok, self.vocab.stoi['<unk>']) for tok in tokens] + \
                      [self.vocab.stoi['<eos>']]
            processed.append(indices)
        
        # Pad to same length
        pad_idx = self.vocab.stoi['<pad>']
        max_len = max(len(seq) for seq in processed)
        padded = []
        for seq in processed:
            padded.append(seq + [pad_idx] * (max_len - len(seq)))
        
        # Convert to tensor (seq_len, batch_size)
        tensor = torch.tensor(padded, dtype=torch.long).t()
        return tensor


def main():
    torch.set_grad_enabled(False)
    args = parse_args()
    model_args = load_object(os.path.join(args.model_path, 'args'))
    
    # Load vocabularies: try to load src_vocab and tgt_vocab separately
    src_vocab_path = os.path.join(args.model_path, 'src_vocab')
    tgt_vocab_path = os.path.join(args.model_path, 'tgt_vocab')
    vocab_path = os.path.join(args.model_path, 'vocab')
    
    # Try to load src_vocab and tgt_vocab
    if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
        # New format: separate vocabularies
        src_vocab = load_object(src_vocab_path)
        tgt_vocab = load_object(tgt_vocab_path)
        print(f"Loaded separate vocabularies: src_vocab ({len(src_vocab)}), tgt_vocab ({len(tgt_vocab)})")
    elif os.path.exists(vocab_path):
        # Old format: single vocabulary (backward compatibility)
        # Load the checkpoint to check embedding sizes
        checkpoint_path = get_model_path(args.model_path, args.epoch)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get embedding sizes from checkpoint
        encoder_embed_size = checkpoint['encoder.embed.weight'].shape[0]
        decoder_embed_size = checkpoint['decoder.embed.weight'].shape[0]
        
        # Load the single vocab (which is tgt_vocab)
        tgt_vocab = load_object(vocab_path)
        
        # Check if we can use tgt_vocab for both src and tgt
        if encoder_embed_size == decoder_embed_size == len(tgt_vocab):
            # Same vocab size for both - can reuse
            src_vocab = tgt_vocab
            print(f"Using single vocabulary for both src and tgt (size: {len(tgt_vocab)})")
        else:
            # Different sizes - we need src_vocab but it's missing
            print(f"ERROR: Vocabulary size mismatch detected!")
            print(f"  Checkpoint encoder embedding size: {encoder_embed_size}")
            print(f"  Checkpoint decoder embedding size: {decoder_embed_size}")
            print(f"  Available vocab ('vocab' file) size: {len(tgt_vocab)}")
            print(f"\nThis checkpoint was trained with separate source and target vocabularies,")
            print(f"but 'src_vocab' file is missing. Please retrain the model with the updated")
            print(f"train.py that saves both src_vocab and tgt_vocab.")
            raise RuntimeError("Cannot load model: src_vocab file missing and vocab sizes don't match")
    else:
        raise FileNotFoundError(f"No vocabulary files found in {args.model_path}")

    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if cuda else 'cpu')
    
    # Create separate field wrappers for src and tgt
    src_field = SimpleField(src_vocab)
    tgt_field = SimpleField(tgt_vocab)
    
    # Create separate metadata for src and tgt
    src_metadata = metadata_factory(model_args, src_vocab)
    tgt_metadata = metadata_factory(model_args, tgt_vocab)

    model = predict_model_factory(model_args, src_metadata, tgt_metadata, get_model_path(args.model_path, args.epoch), tgt_field)
    model = model.to(device)
    model.eval()

    # Read reference file - it's actually just a text file with one sentence per line
    # Not a TSV with question/answer columns
    with open(args.reference_path, 'r', encoding='utf-8') as f:
        ref_answers = [line.strip() for line in f]
    
    # Determine source file path based on reference file
    # For .de reference (German), source is .en (English)
    # For .en reference (English), source is .de (German) - though this is less common
    base_path, ext = args.reference_path.rsplit('.', 1)
    if ext == 'de':
        # Reference is German, source is English
        test_src_path = base_path + '.en'
    elif ext == 'en':
        # Reference is English, source is German (for De->En direction)
        test_src_path = base_path + '.de'
    else:
        raise ValueError(f"Unknown reference file extension: {ext}")
    
    with open(test_src_path, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f]
    
    answers = get_answers(model, questions, args)

    bleu = sacrebleu.corpus_bleu(answers, [ref_answers])

    print(bleu)


if __name__ == '__main__':
    main()
