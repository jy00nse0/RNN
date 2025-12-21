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


def unk_replace(hypothesis, source, attention_weights, unk_token='<unk>', dictionary=None):
    """
    Replace <unk> tokens in hypothesis with source words using attention weights.
    
    Implementation of the UNK replacement technique from Luong et al. (2015)
    "Effective Approaches to Attention-based Neural Machine Translation".
    
    Args:
        hypothesis (str): Generated translation containing <unk> tokens
        source (str): Source sentence (tokenized string)
        attention_weights (torch.Tensor or None): Attention weights of shape (tgt_len, src_len)
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
    
    # Convert attention weights to numpy for easier manipulation
    if isinstance(attention_weights, torch.Tensor):
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
    parser.add_argument('--unk-replace', action='store_true', default=False, 
                        help='Apply UNK replacement using attention weights.')
    return parser.parse_args()


def get_model_path(dir_path, epoch):
    name_start = MODEL_START_FORMAT % epoch
    for path in os.listdir(dir_path):
        if path.startswith(name_start):
            return dir_path + path
    raise ValueError("Model from epoch %d doesn't exist in %s" % (epoch, dir_path))


def get_answers(model, questions, args, return_attention=False):
    batch_size = 1000
    answers = []
    all_attention = [] if return_attention else None
    num_batches = len(questions) // batch_size
    rest = len(questions) % batch_size
    
    for batch in range(num_batches):
        batch_questions = questions[batch * batch_size:(batch + 1) * batch_size]
        if return_attention:
            batch_answers, batch_attention = model(batch_questions,
                                                   sampling_strategy=args.sampling_strategy,
                                                   max_seq_len=args.max_seq_len,
                                                   return_attention=True)
            all_attention.append(batch_attention)
        else:
            batch_answers = model(batch_questions,
                                 sampling_strategy=args.sampling_strategy,
                                 max_seq_len=args.max_seq_len)
        answers.extend(batch_answers)

    if rest != 0:
        batch_questions = questions[-rest:]
        if return_attention:
            batch_answers, batch_attention = model(batch_questions,
                                                   sampling_strategy=args.sampling_strategy,
                                                   max_seq_len=args.max_seq_len,
                                                   return_attention=True)
            all_attention.append(batch_attention)
        else:
            batch_answers = model(batch_questions,
                                 sampling_strategy=args.sampling_strategy,
                                 max_seq_len=args.max_seq_len)
        answers.extend(batch_answers)

    if return_attention:
        # Concatenate all attention weights
        if all_attention:
            attention_weights = torch.cat(all_attention, dim=0)
        else:
            attention_weights = None
        return answers, attention_weights
    
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
    metadata = metadata_factory(model_args, vocab)

    model = predict_model_factory(model_args, metadata, get_model_path(args.model_path + os.path.sep, args.epoch), field)
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
    
    # Get answers with optional attention weights for UNK replacement
    if args.unk_replace:
        answers, attention_weights = get_answers(model, questions, args, return_attention=True)
        
        # Apply UNK replacement to each answer
        if attention_weights is not None:
            replaced_answers = []
            for i, (answer, question) in enumerate(zip(answers, questions)):
                # Get attention weights for this sample
                attn = attention_weights[i]  # (tgt_len, src_len)
                replaced_answer = unk_replace(answer, question, attn)
                replaced_answers.append(replaced_answer)
            answers = replaced_answers
    else:
        answers = get_answers(model, questions, args)

    bleu = sacrebleu.corpus_bleu(answers, [ref_answers])

    print(bleu)


if __name__ == '__main__':
    main()
