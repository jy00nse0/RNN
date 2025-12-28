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
            return dir_path + path
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
    model_args = load_object(args.model_path + os.path.sep + 'args')
    vocab = load_object(args.model_path + os.path.sep + 'vocab')

    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if cuda else 'cpu')
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    if cuda:
        torch.cuda.manual_seed_all(42)

    # Create a simple field wrapper for vocab
    field = SimpleField(vocab)
    # For inference, we need both src and tgt metadata.
    # LIMITATION: During training, only TGT vocab is saved. For same-direction translation
    # (e.g., model trained on en-de and used on en-de), using TGT vocab for both
    # is acceptable IF the source and target vocabularies have the same size.
    # FUTURE WORK: Save both src_vocab and tgt_vocab during training to support
    # different vocab sizes and cross-direction inference.
    tgt_metadata = metadata_factory(model_args, vocab)
    src_metadata = metadata_factory(model_args, vocab)

    model = predict_model_factory(model_args, src_metadata, tgt_metadata, get_model_path(args.model_path + os.path.sep, args.epoch), field)
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
