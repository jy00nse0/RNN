#!/usr/bin/env python
"""
Calculate Perplexity (PPL) for a trained model.

PPL is calculated as: PPL = exp(cross_entropy_loss)
where cross_entropy_loss is the average loss per token on the test set.
"""

import torch
import torch.nn.functional as F
import os
import argparse
import math
from model import train_model_factory
from dataset import dataset_factory, metadata_factory
from serialization import load_object
from constants import MODEL_START_FORMAT
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Script for calculating model perplexity.')
    parser.add_argument('-p', '--model-path', required=True,
                        help='Path to directory with model args, vocabulary and pre-trained pytorch models.')
    parser.add_argument('-e', '--epoch', type=int, required=True,
                        help='Model from this epoch will be loaded.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda if available.')
    parser.add_argument('--dataset', 
                        choices=['twitter-applesupport', 'twitter-amazonhelp', 'twitter-delta',
                                'twitter-spotifycares', 'twitter-uber_support', 'twitter-all',
                                'twitter-small', 'wmt14-en-de', 'wmt15-deen', 'sample100k'],
                        help='Dataset for evaluation.')
    return parser.parse_args()


def get_model_path(dir_path, epoch):
    """Find the model checkpoint file for a given epoch."""
    name_start = MODEL_START_FORMAT % epoch
    for path in os.listdir(dir_path):
        if path.startswith(name_start):
            return os.path.join(dir_path, path)
    raise ValueError(f"Model from epoch {epoch} doesn't exist in {dir_path}")


def evaluate(model, test_iter, metadata, device):
    """
    Evaluate model on test set and return average loss.
    
    Args:
        model: Trained model
        test_iter: Test data iterator
        metadata: Dataset metadata
        device: Device to run evaluation on
    
    Returns:
        Average cross-entropy loss per token
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(test_iter, desc="Evaluating PPL", leave=False):
            question, answer = batch.question, batch.answer
            
            logits = model(question, answer)
            
            # Calculate cross-entropy loss
            # Note: answer[1:] skips the <sos> token since the model predicts
            # starting from the first actual token (teacher forcing during training)
            loss = F.cross_entropy(
                logits.reshape(-1, metadata.vocab_size),
                answer[1:].reshape(-1),
                ignore_index=metadata.padding_idx
            )
            total_loss += loss.item()
    
    return total_loss / len(test_iter)


def main():
    torch.set_grad_enabled(False)
    args = parse_args()
    
    # Load saved model args and vocab
    model_args = load_object(os.path.join(args.model_path, 'args'))
    vocab = load_object(os.path.join(args.model_path, 'vocab'))
    
    # Override dataset if provided
    if args.dataset:
        model_args.dataset = args.dataset
    
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if cuda else 'cpu')
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    if cuda:
        torch.cuda.manual_seed_all(42)
    
    print(f"Loading dataset: {model_args.dataset}")
    
    # Load dataset (we only need test_iter)
    metadata, vocab, train_iter, val_iter, test_iter = dataset_factory(model_args, device)
    
    # Create model
    model = train_model_factory(model_args, metadata)
    
    # Load checkpoint
    checkpoint_path = get_model_path(args.model_path, args.epoch)
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle DataParallel models
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    test_loss = evaluate(model, test_iter, metadata, device)
    
    # Calculate perplexity
    ppl = math.exp(test_loss)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Perplexity (PPL): {ppl:.4f}")
    
    # Return PPL in a format that can be easily parsed
    print(f"PPL = {ppl:.4f}")


if __name__ == '__main__':
    main()
