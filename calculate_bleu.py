#!/usr/bin/env python

import torch
import pandas as pd
import os
import argparse
import subprocess
from model import predict_model_factory
from dataset import field_factory, metadata_factory
from serialization import load_object
from constants import MODEL_START_FORMAT


class SimpleField:
    """Simple field-like object that wraps vocab for compatibility"""
    def __init__(self, vocab, add_sos=False, add_eos=True):
        """
        Args:
            vocab: Vocabulary object
            add_sos: Whether to add <sos> token when processing
            add_eos: Whether to add <eos> token when processing
        """
        self.vocab = vocab
        self.add_sos = add_sos
        self.add_eos = add_eos
    
    def preprocess(self, text):
        """Tokenize text by splitting on whitespace"""
        return text.strip().split()
    
    def process(self, batch):
        """Convert list of token lists to tensor"""
        # Process tokens to indices
        processed = []
        for tokens in batch:
            indices = []
            if self.add_sos:
                indices.append(self.vocab.stoi['<sos>'])
            indices.extend([self.vocab.stoi.get(tok, self.vocab.stoi['<unk>']) for tok in tokens])
            if self.add_eos:
                indices.append(self.vocab.stoi['<eos>'])
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
    parser.add_argument('--lowercase', action='store_true', default=False, help='Lowercase for BLEU evaluation.')
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


def calculate_bleu_with_perl(hypotheses, reference_path, lowercase=False):
    """
    Calculate BLEU score using multi-bleu.perl script.
    
    Args:
        hypotheses: List of hypothesis strings
        reference_path: Path to reference file
        lowercase: Whether to use lowercase comparison
    
    Returns:
        BLEU score string from multi-bleu.perl
    """
    script_path = os.path.join(os.path.dirname(__file__), 'multi-bleu.perl')
    
    # Prepare command
    cmd = ['perl', script_path]
    if lowercase:
        cmd.append('-lc')
    cmd.append(reference_path)
    
    # Run multi-bleu.perl with hypotheses as stdin
    hypotheses_text = '\n'.join(hypotheses) + '\n'
    
    try:
        result = subprocess.run(
            cmd,
            input=hypotheses_text,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running multi-bleu.perl: {e}")
        print(f"stderr: {e.stderr}")
        raise


def main():
    torch.set_grad_enabled(False)
    args = parse_args()
    model_args = load_object(os.path.join(args.model_path, 'args'))
    
    # Load vocabularies with backward compatibility
    src_vocab_path = os.path.join(args.model_path, 'src_vocab')
    tgt_vocab_path = os.path.join(args.model_path, 'tgt_vocab')
    legacy_vocab_path = os.path.join(args.model_path, 'vocab')
    
    src_vocab = None
    tgt_vocab = None
    
    if os.path.exists(src_vocab_path):
        src_vocab = load_object(src_vocab_path)
    
    if os.path.exists(tgt_vocab_path):
        tgt_vocab = load_object(tgt_vocab_path)
    
    # Fallback to legacy vocab if either is missing
    if src_vocab is None or tgt_vocab is None:
        missing_paths = []
        if src_vocab is None:
            missing_paths.append(src_vocab_path)
        if tgt_vocab is None:
            missing_paths.append(tgt_vocab_path)
        print(f"Warning: Separate src_vocab/tgt_vocab missing at: {', '.join(missing_paths)}. Filling missing vocab(s) from legacy single vocab.")
        
        if os.path.exists(legacy_vocab_path):
            legacy_vocab = load_object(legacy_vocab_path)
            if src_vocab is None:
                src_vocab = legacy_vocab
            if tgt_vocab is None:
                tgt_vocab = legacy_vocab
        else:
            raise FileNotFoundError("No vocabulary files found (src_vocab, tgt_vocab, or vocab)")

    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if cuda else 'cpu')
    

    # Create separate fields for source and target
    # Source: only add <eos> (no <sos>), matching training behavior
    # Target: used only for decoding, SOS/EOS indices handled by Seq2SeqPredict
    src_field = SimpleField(src_vocab, add_sos=False, add_eos=True)
    tgt_field = SimpleField(tgt_vocab, add_sos=False, add_eos=False)
    
    # Use corresponding vocabularies for metadata creation, with backward compatibility fallback.
    tgt_metadata = metadata_factory(model_args, tgt_vocab)
    src_metadata = metadata_factory(model_args, src_vocab)

    model = predict_model_factory(model_args, src_metadata, tgt_metadata, get_model_path(args.model_path, args.epoch), src_field, tgt_field)
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

    # Use multi-bleu.perl for BLEU calculation
    bleu_output = calculate_bleu_with_perl(answers, args.reference_path, lowercase=args.lowercase)
    print(bleu_output)


if __name__ == '__main__':
    main()
