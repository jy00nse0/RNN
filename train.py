#!/usr/bin/env python3

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from dataset import dataset_factory
from model import train_model_factory
from serialization import save_object, save_model, save_vocab
from datetime import datetime
from util import embedding_size_from_name

"""
[Revised] train.py for RNN Paper Reproduction
Changes:
1. Optimizer: Adam -> SGD (Paper Sec 4.1)
2. Scheduler: Added Halving Schedule (Paper Sec 4.1)
   - Base model: Halve after epoch 5
   - Dropout model: Halve after epoch 8
3. Defaults: Updated batch_size(128), lr(1.0) to match paper.
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Script for training seq2seq chatbot.')
    # [Paper] Total Epochs: 10 (Base) or 12 (Dropout)
    parser.add_argument('--max-epochs', type=int, default=10, help='Max number of epochs models will be trained.')
    # [Paper] Gradient Clipping: Norm > 5
    parser.add_argument('--gradient-clip', type=float, default=5.0, help='Gradient clip value.')
    # [Paper] Batch Size: 128
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size.')
    # [Paper] Initial Learning Rate: 1.0 (SGD)
    parser.add_argument('--learning-rate', type=float, default=1.0, help='Initial learning rate.')
    # [Paper] LR Decay Start: 5 (Base) or 8 (Dropout)
    parser.add_argument('--lr-decay-start', type=int, default=5, 
                        help='Epoch after which to start halving learning rate. (Base: 5, Dropout: 8)')
    
    parser.add_argument('--train-embeddings', action='store_true',
                        help='Should gradients be propagated to word embeddings.')
    parser.add_argument('--embedding-type', type=str, default=None)
    parser.add_argument('--save-path', default='.save',
                        help='Folder where models (and other configs) will be saved during training.')
    parser.add_argument('--save-every-epoch', action='store_true',
                        help='Save model every epoch regardless of validation loss.')
    parser.add_argument('--dataset', choices=['twitter-applesupport', 'twitter-amazonhelp', 'twitter-delta',
                                              'twitter-spotifycares', 'twitter-uber_support', 'twitter-all',
                                              'twitter-small'],
                        help='Dataset for training model.')
    parser.add_argument('--teacher-forcing-ratio', type=float, default=0.5,
                        help='Teacher forcing ratio used in seq2seq models. [0-1]')
   # [Paper] Experimental Setup
    parser.add_argument('--reverse', action='store_true', 
                        help='[Experiment] Reverse source sequence (excluding special tokens) for LSTM inputs.')
    # cuda
    gpu_args = parser.add_argument_group('GPU', 'GPU related settings.')
    gpu_args.add_argument('--cuda', action='store_true', default=False, help='Use cuda if available.')
    gpu_args.add_argument('--multi-gpu', action='store_true', default=False, help='Use multiple GPUs if available.')

    # embeddings hyperparameters
    embeddings = parser.add_mutually_exclusive_group()
    # Model Hyperparameters
    encoder_args = parser.add_argument_group('Encoder', 'Encoder hyperparameters.')

    encoder_args.add_argument('--encoder-rnn-cell', choices=['LSTM', 'GRU'], default='LSTM',
                              help='Encoder RNN cell type.')
    encoder_args.add_argument('--encoder-hidden-size', type=int, default=1000, help='Encoder RNN hidden size.')
    encoder_args.add_argument('--encoder-num-layers', type=int, default=4, help='Encoder RNN number of layers.')
    encoder_args.add_argument('--encoder-rnn-dropout', type=float, default=0.0, help='Encoder RNN dropout probability.')
    encoder_args.add_argument('--encoder-bidirectional', action='store_true', help='Use bidirectional encoder.')

    # decoder hyperparameters
    decoder_args = parser.add_argument_group('Decoder', 'Decoder hyperparameters.')
    decoder_args.add_argument('--decoder-type', choices=['bahdanau', 'luong'], default='luong',
                              help='Type of the decoder.')
    decoder_args.add_argument('--decoder-rnn-cell', choices=['LSTM', 'GRU'], default='LSTM',
                              help='Decoder RNN cell type.')
    decoder_args.add_argument('--decoder-hidden-size', type=int, default=1000, help='Decoder RNN hidden size.')
    decoder_args.add_argument('--decoder-num-layers', type=int, default=4, help='Decoder RNN number of layers.')
    decoder_args.add_argument('--decoder-rnn-dropout', type=float, default=0.0, help='Decoder RNN dropout probability.')
    decoder_args.add_argument('--luong-attn-hidden-size', type=int, default=1000,
                              help='Luong decoder attention hidden projection size')
    decoder_args.add_argument('--luong-input-feed', action='store_true',
                              help='Whether Luong decoder should use input feeding approach.')
    decoder_args.add_argument('--decoder-init-type', choices=['zeros', 'bahdanau', 'adjust_pad', 'adjust_all'],
                              default='zeros', help='Decoder initial RNN hidden state initialization.')

    # attention hyperparameters
    attention_args = parser.add_argument_group('Attention', 'Attention hyperparameters.')
    attention_args.add_argument('--attention-type', choices=['none', 'global', 'local-m', 'local-p'], default='global',
                                help='Attention type.')
    attention_args.add_argument('--attention-score', choices=['dot', 'general', 'concat', 'location'], default='dot',
                                help='Attention score function type.')
    # [Paper] Window size D=10
    attention_args.add_argument('--half-window-size', type=int, default=10,
                                help='D parameter from Luong et al. paper. Used only for local attention.')
    attention_args.add_argument('--local-p-hidden-size', type=int, default=1000,
                                help='Local-p attention hidden size (used when predicting window position).')
    attention_args.add_argument('--concat-attention-hidden-size', type=int, default=1000,
                                help='Attention layer hidden size. Used only with concat score function.')

    args = parser.parse_args()

    # [Note] Embeddings logic preserved but likely not used if training from scratch with vocab 50k
    if not args.embedding_type and not args.embedding_size:
        args.embedding_size = 1000 # Paper default

    args.save_path += os.path.sep + datetime.now().strftime("%Y-%m-%d-%H:%M")

    print(args)
    return args

def batch_reverse_source(src_tensor, pad_idx):
    """
    [New] 소스 텐서를 뒤집는 함수 (Memory-efficient in-place flip attempt or clone)
    Input: (seq_len, batch)
    Assumes structure: <sos> w1 w2 ... wn <eos> <pad> ...
    Target: <sos> wn ... w2 w1 <eos> <pad> ...
    """
    # Clone to avoid modifying original data if needed, or modify in place
    rev_src = src_tensor.clone()
    seq_len, batch_size = src_tensor.size()

    for b in range(batch_size):
        # Find length of current sequence (excluding padding)
        # Assuming padding is at the end.
        non_pad_mask = (src_tensor[:, b] != pad_idx)
        valid_len = non_pad_mask.sum().item()
        
        if valid_len <= 2: # Only <sos> and <eos> or empty
            continue
            
        # Indices to reverse: from 1 to valid_len-2 (inclusive)
        # Index 0 is <sos>, Index valid_len-1 is <eos>
        # Reverse the content: src[1 : valid_len-1]
        content = src_tensor[1 : valid_len-1, b]
        rev_src[1 : valid_len-1, b] = torch.flip(content, dims=[0])
        
    return rev_src
   
def evaluate(model, val_iter, metadata,reverse_src=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_iter:
            question, answer = batch.question, batch.answer
            # [Feature] Reverse Source if flag is set
            if reverse_src:
                question = batch_reverse_source(question, metadata.padding_idx)
            logits = model(question, answer)
            loss = F.cross_entropy(logits.view(-1, metadata.vocab_size), answer[1:].view(-1),
                                   ignore_index=metadata.padding_idx)
            total_loss += loss.item()
    return total_loss / len(val_iter)


def train(model, optimizer, train_iter, metadata, grad_clip,reverse_src=False):
    model.train()
    total_loss = 0
    for batch in train_iter:
        question, answer = batch.question, batch.answer
        # [Feature] Reverse Source if flag is set
        if reverse_src:
            question = batch_reverse_source(question, metadata.padding_idx)
        logits = model(question, answer)

        optimizer.zero_grad()
        loss = F.cross_entropy(logits.view(-1, metadata.vocab_size), answer[1:].view(-1),
                               ignore_index=metadata.padding_idx)
        loss.backward()

        total_loss += loss.item()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return total_loss / len(train_iter)


def adjust_learning_rate(optimizer, epoch, decay_start):
    """
    [New] Halve learning rate every epoch after decay_start epoch.
    Corresponds to Paper Section 4.1.
    """
    if epoch >= decay_start:
        for param_group in optimizer.param_groups:
            # param_group['lr'] = param_group['lr'] * 0.5 # This would compound 0.5 every time called
            # Since this function is called once per epoch loop, simple multiplication is fine.
            # However, safer way is to check if we just crossed the boundary or do it incrementally.
            # Given the loop structure below, we can just multiply by 0.5 at the start of qualifying epochs.
            param_group['lr'] *= 0.5
            print(f"Decaying learning rate to {param_group['lr']}")


def main():
    args = parse_args()
    cuda = torch.cuda.is_available() and args.cuda
    torch.set_default_tensor_type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)
    device = torch.device('cuda' if cuda else 'cpu')

    print("Using %s for training" % ('GPU' if cuda else 'CPU'))
    print('Loading dataset...', end='', flush=True)
    metadata, vocab, train_iter, val_iter, test_iter = dataset_factory(args, device)
    print('Done.')

    print('Saving vocab and args...', end='')
    save_vocab(vocab, args.save_path + os.path.sep + 'vocab')
    save_object(args, args.save_path + os.path.sep + 'args')
    print('Done')

    model = train_model_factory(args, metadata)
    if cuda and args.multi_gpu:
        model = nn.DataParallel(model, dim=1)
    
    # [Check] Initialization is handled inside model.py -> util.init_weights
    print(model)

    # [Revised] Optimizer: SGD as per paper
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    try:
        best_val_loss = None
        for epoch in range(args.max_epochs):
            start = datetime.now()
            
            # [Revised] LR Scheduling
            # 논문: "after 5 epochs, we begin to halve the learning rate every epoch"
            # epoch is 0-indexed here. 
            # If decay_start=5, it means after epoch 4 (0,1,2,3,4 finished). 
            # So if epoch >= 5, we decay.
            if epoch >= args.lr_decay_start:
                adjust_learning_rate(optimizer, epoch, args.lr_decay_start)

            train_loss = train(model, optimizer, train_iter, metadata, args.gradient_clip)
            val_loss = evaluate(model, val_iter, metadata)
            print("[Epoch=%d/%d] train_loss %f - val_loss %f time=%s " %
                  (epoch + 1, args.max_epochs, train_loss, val_loss, datetime.now() - start), end='')

            if args.save_every_epoch or not best_val_loss or val_loss < best_val_loss:
                print('(Saving model...', end='')
                save_model(args.save_path, model, epoch + 1, train_loss, val_loss)
                print('Done)', end='')
                best_val_loss = val_loss
            print()
            
    except (KeyboardInterrupt, BrokenPipeError):
        print('[Ctrl-C] Training stopped.')

    test_loss = evaluate(model, test_iter, metadata)
    print("Test loss %f" % test_loss)


if __name__ == '__main__':
    main()
