#!/usr/bin/env python3

import torch
import torch.nn as nn
import os
import argparse
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


class ModelDecorator(nn.Module):
    """
    Simple decorator around Seq2SeqPredict model which packs input question in list and unpacks output list into single
    answer. This allows client to have simple interface for dialog-like model (doesn't need to worry about wrapping and
    unwrapping).
    """

    def __init__(self, model):
        super(ModelDecorator, self).__init__()
        self.model = model

    def forward(self, question, sampling_strategy, max_seq_len):
        return self.model([question], sampling_strategy, max_seq_len)[0]


customer_service_models = {
    'apple': ('pretrained-models/apple', 39),
    'amazon': ('pretrained-models/amazon', 10),
    'uber': ('pretrained-models/uber', 58),
    'delta': ('pretrained-models/delta', 44),
    'spotify': ('pretrained-models/spotify', 14)
}


def parse_args():
    parser = argparse.ArgumentParser(description='Script for "talking" with pre-trained chatbot.')
    parser.add_argument('-cs', '--customer-service', choices=['apple', 'amazon', 'uber', 'delta', 'spotify'])
    parser.add_argument('-p', '--model-path',
                        help='Path to directory with model args, vocabulary and pre-trained pytorch models.')
    parser.add_argument('-e', '--epoch', type=int, help='Model from this epoch will be loaded.')
    parser.add_argument('--sampling-strategy', choices=['greedy', 'random', 'beam_search'], default='greedy',
                        help='Strategy for sampling output sequence.')
    parser.add_argument('--max-seq-len', type=int, default=50, help='Maximum length for output sequence.')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use cuda if available.')

    args = parser.parse_args()

    if args.customer_service:
        cs = customer_service_models[args.customer_service]
        args.model_path = cs[0]
        args.epoch = cs[1]

    return args


def get_model_path(dir_path, epoch):
    name_start = MODEL_START_FORMAT % epoch
    for path in os.listdir(dir_path):
        if path.startswith(name_start):
            return os.path.join(dir_path, path)
    raise ValueError("Model from epoch %d doesn't exist in %s" % (epoch, dir_path))


def main():
    torch.set_grad_enabled(False)
    args = parse_args()
    print('Args loaded')
    model_args = load_object(os.path.join(args.model_path, 'args'))
    print('Model args loaded.')
    
    # Load vocabularies with backward compatibility
    src_vocab_path = os.path.join(args.model_path, 'src_vocab')
    tgt_vocab_path = os.path.join(args.model_path, 'tgt_vocab')
    legacy_vocab_path = os.path.join(args.model_path, 'vocab')
    
    src_vocab = None
    tgt_vocab = None
    
    if os.path.exists(src_vocab_path):
        src_vocab = load_object(src_vocab_path)
        print('Source vocab loaded.')
    
    if os.path.exists(tgt_vocab_path):
        tgt_vocab = load_object(tgt_vocab_path)
        print('Target vocab loaded.')
    
    # Fallback to legacy vocab if either is missing
    if src_vocab is None or tgt_vocab is None:
        if os.path.exists(legacy_vocab_path):
            print('Using legacy single vocab for missing vocabularies.')
            legacy_vocab = load_object(legacy_vocab_path)
            if src_vocab is None:
                src_vocab = legacy_vocab
            if tgt_vocab is None:
                tgt_vocab = legacy_vocab
        else:
            raise FileNotFoundError("No vocabulary files found (src_vocab, tgt_vocab, or vocab)")

    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if cuda else 'cpu')

    
    print("Using %s for inference" % ('GPU' if cuda else 'CPU'))

    # Create separate fields for source and target
    # Source: only add <eos> (no <sos>), matching training behavior
    # Target: used only for decoding, SOS/EOS indices handled by Seq2SeqPredict
    src_field = SimpleField(src_vocab, add_sos=False, add_eos=True)
    tgt_field = SimpleField(tgt_vocab, add_sos=False, add_eos=False)
    
    tgt_metadata = metadata_factory(model_args, tgt_vocab)
    src_metadata = metadata_factory(model_args, src_vocab)

    model = ModelDecorator(
        predict_model_factory(model_args, src_metadata, tgt_metadata, get_model_path(args.model_path, args.epoch), src_field, tgt_field))
    model = model.to(device)
    print('model loaded')
    model.eval()

    question = ''
    print('\n\nBot: Hi, how can I help you?', flush=True)
    while question != 'bye':
        while True:
            print('Me: ', end='')
            question = input()
            if question:
                break

        response = model(question, sampling_strategy=args.sampling_strategy, max_seq_len=args.max_seq_len)
        print('Bot: ' + response)


if __name__ == '__main__':
    main()
