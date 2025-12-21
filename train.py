#!/usr/bin/env python3

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from dataset import dataset_factory
from model import train_model_factory
from serialization import save_object, save_model, save_vocab
from datetime import datetime
from util import embedding_size_from_name
from tqdm import tqdm
import math

"""
[Optimized] train.py for RNN Paper Reproduction

Original Changes:
1. Optimizer: Adam -> SGD (Paper Sec 4.1)
2. Scheduler: Added Halving Schedule (Paper Sec 4.1)
   - Base model: Halve after epoch 5
   - Dropout model: Halve after epoch 8
3. Defaults: Updated batch_size(128), lr(1.0) to match paper.

New Optimizations:
1. view() -> reshape() for safer tensor operations
2. AMP (Mixed Precision) support with --amp flag
3. DataLoader workers configuration
4. zero_grad(set_to_none=True) for better performance
5. TF32 and cuDNN benchmarking enabled

Real-time Monitoring:
1. Gradient monitoring (norm, NaN, Inf, extrema)
2. Loss monitoring (spikes, divergence)
3. Activation monitoring (hidden states, attention)
4. Sample generation checks
"""


def monitor_gradients(model):
    """Gradient 통계를 계산하여 학습 붕괴 감지"""
    total_norm = 0.0
    grad_stats = {
        'max_grad': 0.0,
        'min_grad': float('inf'),
        'nan_count': 0,
        'inf_count': 0,
        'zero_count': 0
    }
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            
            grad_stats['max_grad'] = max(grad_stats['max_grad'], p.grad.abs().max().item())
            grad_stats['min_grad'] = min(grad_stats['min_grad'], p.grad.abs().min().item())
            grad_stats['nan_count'] += torch.isnan(p.grad).sum().item()
            grad_stats['inf_count'] += torch.isinf(p.grad).sum().item()
            grad_stats['zero_count'] += (p.grad == 0).sum().item()
    
    grad_stats['total_norm'] = total_norm ** 0.5
    return grad_stats


class LossMonitor:
    """Loss의 급격한 변화를 감지"""
    def __init__(self, window_size=10, spike_threshold=3.0):
        self.losses = []
        self.window_size = window_size
        self.spike_threshold = spike_threshold
    
    def add(self, loss):
        self.losses.append(loss)
        if len(self.losses) > self.window_size:
            self.losses.pop(0)
    
    def is_spiking(self):
        """Loss가 급증했는지 확인"""
        if len(self.losses) < 2:
            return False
        
        recent_avg = sum(self.losses[-3:]) / min(3, len(self.losses))
        older_avg = sum(self.losses[:-3]) / max(1, len(self.losses) - 3)
        
        if older_avg > 0:
            spike_ratio = recent_avg / older_avg
            return spike_ratio > self.spike_threshold
        return False
    
    def is_diverging(self):
        """Loss가 발산하는지 확인"""
        if len(self.losses) < self.window_size:
            return False
        
        increasing = all(self.losses[i] < self.losses[i+1] 
                        for i in range(-self.window_size, -1))
        return increasing


def calculate_perplexity(loss):
    """Loss를 Perplexity로 변환 (더 직관적)"""
    return math.exp(loss)


class ActivationMonitor:
    """RNN hidden states와 attention weights 모니터링"""
    def __init__(self):
        self.stats = {}
    
    def register_hooks(self, model):
        """Model에 forward hook 등록"""
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                
                self.stats[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'max': output.max().item(),
                    'min': output.min().item(),
                    'nan_count': torch.isnan(output).sum().item(),
                    'inf_count': torch.isinf(output).sum().item(),
                }
            return hook
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.LSTM, nn.GRU)):
                module.register_forward_hook(hook_fn(name))
    
    def check_anomalies(self):
        """활성화 값의 이상 징후 감지"""
        warnings = []
        for name, stat in self.stats.items():
            if stat['nan_count'] > 0:
                warnings.append(f"⚠️  {name}: NaN detected!")
            if stat['inf_count'] > 0:
                warnings.append(f"⚠️  {name}: Inf detected!")
            if abs(stat['mean']) > 100:
                warnings.append(f"⚠️  {name}: Large mean ({stat['mean']:.2f})")
            if stat['std'] < 1e-5:
                warnings.append(f"⚠️  {name}: Dying activations (std={stat['std']:.2e})")
        return warnings


def sample_generation(model, val_iter, metadata, reverse_src=False, num_samples=3):
    """실제 생성 결과를 확인하여 붕괴 감지"""
    model.eval()
    
    with torch.no_grad():
        batch = next(iter(val_iter))
        question, answer = batch.question[:, :num_samples], batch.answer[:, :num_samples]
        
        # Check if model has greedy_decode method
        if hasattr(model, 'module'):
            actual_model = model.module
        else:
            actual_model = model
            
        if not hasattr(actual_model, 'greedy_decode'):
            print("\n⚠️  Model does not have greedy_decode method. Skipping sample generation.")
            return
        
        # Greedy decoding
        output = actual_model.greedy_decode(question, max_len=50)
        
        print("\n" + "="*70)
        print("Sample Generations:")
        print("="*70)
        
        for i in range(num_samples):
            src = [metadata.vocab.itos[idx] for idx in question[:, i].cpu().numpy() if idx != metadata.padding_idx]
            tgt = [metadata.vocab.itos[idx] for idx in answer[:, i].cpu().numpy() if idx != metadata.padding_idx]
            out = [metadata.vocab.itos[idx] for idx in output[:, i].cpu().numpy() if idx != metadata.padding_idx]
            
            print(f"\nSample {i+1}:")
            print(f"  Input:   {' '.join(src)}")
            print(f"  Target:  {' '.join(tgt)}")
            print(f"  Output:  {' '.join(out)}")
            
            if len(set(out)) < 3:
                print(f"  ⚠️  WARNING: Repetitive output detected!")
            
            if all(w == out[0] for w in out):
                print(f"  🚨 CRITICAL: Model generating same token!")
        
        print("="*70)


def parse_args():
    parser = argparse.ArgumentParser(description='Script for training seq2seq chatbot.')
    
    # ===== Paper Hyperparameters =====
    # [Paper] Total Epochs: 10 (Base) or 12 (Dropout)
    parser.add_argument('--max-epochs', type=int, default=10, 
                       help='Max number of epochs models will be trained.')
    # [Paper] Gradient Clipping: Norm > 5
    parser.add_argument('--gradient-clip', type=float, default=5.0, 
                       help='Gradient clip value.')
    # [Paper] Batch Size: 128
    parser.add_argument('--batch-size', type=int, default=128, 
                       help='Batch size.')
    # [Paper] Initial Learning Rate: 1.0 (SGD)
    parser.add_argument('--learning-rate', type=float, default=1.0, 
                       help='Initial learning rate.')
    # [Paper] LR Decay Start: 5 (Base) or 8 (Dropout)
    parser.add_argument('--lr-decay-start', type=int, default=5, 
                       help='Epoch after which to start halving learning rate. (Base: 5, Dropout: 8)')
    
    # ===== Training Configuration =====
    parser.add_argument('--train-embeddings', action='store_true',
                       help='Should gradients be propagated to word embeddings.')
    parser.add_argument('--embedding-type', type=str, default=None)
    parser.add_argument('--save-path', default='.save',
                       help='Folder where models (and other configs) will be saved during training.')
    parser.add_argument('--save-every-epoch', action='store_true',
                       help='Save model every epoch regardless of validation loss.')
    parser.add_argument('--dataset', 
                       choices=['twitter-applesupport', 'twitter-amazonhelp', 'twitter-delta',
                               'twitter-spotifycares', 'twitter-uber_support', 'twitter-all',
                               'twitter-small', 'wmt14-en-de', 'wmt15-deen', 'sample100k'],
                       help='Dataset for training model.')
    parser.add_argument('--teacher-forcing-ratio', type=float, default=1.0,
                       help='Teacher forcing ratio used in seq2seq models. [0-1]')
    
    # [Paper] Experimental Setup
    parser.add_argument('--reverse', action='store_true', 
                       help='[Experiment] Reverse source sequence (excluding special tokens) for LSTM inputs.')
    
    # ===== GPU Settings =====
    gpu_args = parser.add_argument_group('GPU', 'GPU related settings.')
    gpu_args.add_argument('--cuda', action='store_true', default=False, 
                         help='Use cuda if available.')
    gpu_args.add_argument('--multi-gpu', action='store_true', default=False, 
                         help='Use multiple GPUs if available.')
    
    # ===== [OPTIMIZED] Performance Options =====
    perf_args = parser.add_argument_group('Performance', 'Performance optimization settings.')
    perf_args.add_argument('--num-workers', type=int, default=12,
                          help='Number of DataLoader workers (0=single process, 4-8 recommended).')
    perf_args.add_argument('--amp', action='store_true',default=True,
                          help='Use Automatic Mixed Precision (1.5-2x faster but may affect reproducibility).')

    # ===== Embedding Hyperparameters =====
    parser.add_argument('--embedding-size', type=int, default=None, 
                       help='Embedding size.')
    
    # ===== Encoder Hyperparameters =====
    encoder_args = parser.add_argument_group('Encoder', 'Encoder hyperparameters.')
    encoder_args.add_argument('--encoder-rnn-cell', choices=['LSTM', 'GRU'], default='LSTM',
                             help='Encoder RNN cell type.')
    encoder_args.add_argument('--encoder-hidden-size', type=int, default=1000, 
                             help='Encoder RNN hidden size.')
    encoder_args.add_argument('--encoder-num-layers', type=int, default=4, 
                             help='Encoder RNN number of layers.')
    encoder_args.add_argument('--encoder-rnn-dropout', type=float, default=0.0, 
                             help='Encoder RNN dropout probability.')
    encoder_args.add_argument('--encoder-bidirectional', action='store_true', 
                             help='Use bidirectional encoder.')

    # ===== Decoder Hyperparameters =====
    decoder_args = parser.add_argument_group('Decoder', 'Decoder hyperparameters.')
    decoder_args.add_argument('--decoder-type', choices=['bahdanau', 'luong'], default='luong',
                             help='Type of the decoder.')
    decoder_args.add_argument('--decoder-rnn-cell', choices=['LSTM', 'GRU'], default='LSTM',
                             help='Decoder RNN cell type.')
    decoder_args.add_argument('--decoder-hidden-size', type=int, default=1000, 
                             help='Decoder RNN hidden size.')
    decoder_args.add_argument('--decoder-num-layers', type=int, default=4, 
                             help='Decoder RNN number of layers.')
    decoder_args.add_argument('--decoder-rnn-dropout', type=float, default=0.0, 
                             help='Decoder RNN dropout probability.')
    decoder_args.add_argument('--luong-attn-hidden-size', type=int, default=1000,
                             help='Luong decoder attention hidden projection size')
    decoder_args.add_argument('--luong-input-feed', action='store_true',
                             help='Whether Luong decoder should use input feeding approach.')
    decoder_args.add_argument('--decoder-init-type', 
                             choices=['zeros', 'bahdanau', 'adjust_pad', 'adjust_all'],
                             default='zeros', 
                             help='Decoder initial RNN hidden state initialization.')

    # ===== Attention Hyperparameters =====
    attention_args = parser.add_argument_group('Attention', 'Attention hyperparameters.')
    attention_args.add_argument('--attention-type', 
                               choices=['none', 'global', 'local-m', 'local-p'], 
                               default='global',
                               help='Attention type.')
    attention_args.add_argument('--attention-score', 
                               choices=['dot', 'general', 'concat', 'location'], 
                               default='dot',
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
        args.embedding_size = 1000  # Paper default

    args.save_path += os.path.sep + datetime.now().strftime("%Y-%m-%d-%H:%M")

    print(args)
    return args


def batch_reverse_source(src_tensor, pad_idx, batch_first=False):
    """
    Reverse the content of source sequences while preserving <sos>, <eos>, and <pad> positions.
    
    Supports both (seq_len, batch) and (batch, seq_len) tensor shapes.
    Handles padding at both beginning (left) and end (right) of sequences.
    
    Args:
        src_tensor: Input tensor of shape (seq_len, batch) or (batch, seq_len)
        pad_idx: Index of padding token
        batch_first: If True, input shape is (batch, seq_len); if False, (seq_len, batch)
    
    Input structure: <sos> w1 w2 ... wn <eos> (with optional <pad> before or after)
    Output structure: <sos> wn ... w2 w1 <eos> (with same <pad> positions preserved)
    
    Returns:
        Tensor with reversed content, same shape as input
    """
    rev_src = src_tensor.clone()
    
    if batch_first:
        batch_size, seq_len = src_tensor.size()
    else:
        seq_len, batch_size = src_tensor.size()
    
    for b in range(batch_size):
        if batch_first:
            seq = src_tensor[b, :]
        else:
            seq = src_tensor[:, b]
        
        # Find indices of non-padding tokens
        non_pad_indices = (seq != pad_idx).nonzero(as_tuple=True)[0]
        
        if len(non_pad_indices) <= 2:
            # Only <sos> and <eos> or fewer - nothing to reverse
            continue
        
        # First and last non-padding positions are <sos> and <eos>
        first_non_pad = non_pad_indices[0].item()
        last_non_pad = non_pad_indices[-1].item()
        
        # Content to reverse: tokens between <sos> (first_non_pad) and <eos> (last_non_pad)
        start_idx = first_non_pad + 1  # skip <sos>
        end_idx = last_non_pad  # up to but not including <eos>
        
        if end_idx <= start_idx:
            # No content to reverse (only special tokens)
            continue
        
        # Extract content and reverse
        if batch_first:
            content = src_tensor[b, start_idx:end_idx]
            rev_src[b, start_idx:end_idx] = torch.flip(content, dims=[0])
        else:
            content = src_tensor[start_idx:end_idx, b]
            rev_src[start_idx:end_idx, b] = torch.flip(content, dims=[0])
    
    return rev_src


def evaluate(model, val_iter, metadata, reverse_src=False):
    """
    [OPTIMIZED] Evaluate model on validation set.
    - Changed view() to reshape() for safer tensor operations.
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_iter, desc="Evaluating", leave=False):
            question, answer = batch.question, batch.answer
            
            # [Feature] Reverse Source if flag is set
            # if reverse_src:
            #     question = batch_reverse_source(question, metadata.padding_idx)
            
            logits = model(question, answer)
            
            # [OPTIMIZED] reshape() instead of view() - safer, handles non-contiguous tensors
            loss = F.cross_entropy(
                logits.reshape(-1, metadata.vocab_size),
                answer[1:].reshape(-1),
                ignore_index=metadata.padding_idx
            )
            total_loss += loss.item()
    
    return total_loss / len(val_iter)


def train(model, optimizer, train_iter, metadata, grad_clip, reverse_src=False, 
          use_amp=False, scaler=None, epoch=0):
    """
    [OPTIMIZED] Train model for one epoch with real-time monitoring.
    
    Optimizations:
    - AMP (Mixed Precision) support
    - reshape() instead of view()
    - zero_grad(set_to_none=True) for better memory efficiency
    - TF32 and cuDNN benchmarking enabled
    
    Real-time Monitoring:
    - Gradient monitoring (norm, NaN, Inf)
    - Loss monitoring (spikes, divergence)
    - Activation monitoring (every 100 batches)
    - Batch-level statistics (every 100 batches)
    
    Args:
        model: Neural network model
        optimizer: Optimizer instance
        train_iter: Training data iterator
        metadata: Dataset metadata
        grad_clip: Gradient clipping value
        reverse_src: Whether to reverse source sequences
        use_amp: Whether to use automatic mixed precision
        scaler: GradScaler for AMP (required if use_amp=True)
        epoch: Current epoch number (for logging)
    
    Returns:
        Average training loss for the epoch
    """
    # [OPTIMIZED] Enable performance optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    model.train()
    
    # Initialize monitoring tools
    loss_monitor = LossMonitor(window_size=10, spike_threshold=3.0)
    activation_monitor = ActivationMonitor()
    activation_monitor.register_hooks(model)
    
    total_loss = 0
    batch_losses = []
    grad_norms = []
    batch_idx = 0
    
    for batch in tqdm(train_iter, desc=f"Training Epoch {epoch + 1}", leave=False):
        question, answer = batch.question, batch.answer
        
        # [Feature] Reverse Source if flag is set
        # if reverse_src:
        #     question = batch_reverse_source(question, metadata.padding_idx)
        
        # [OPTIMIZED] AMP context (no-op if use_amp=False)
        with autocast(enabled=use_amp):
            logits = model(question, answer)
            
            # [OPTIMIZED] reshape() instead of view()
            loss = F.cross_entropy(
                logits.reshape(-1, metadata.vocab_size),
                answer[1:].reshape(-1),
                ignore_index=metadata.padding_idx
            )
        
        # Critical: Check for NaN/Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f"🚨 TRAINING COLLAPSED: Loss is {loss.item()} at batch {batch_idx}")
        
        # Add loss to monitor
        loss_monitor.add(loss.item())
        batch_losses.append(loss.item())
        
        # [OPTIMIZED] AMP-aware backward and optimizer step
        if use_amp:
            # Scale loss and backward
            scaler.scale(loss).backward()
            
            # Unscale before gradient clipping (CRITICAL!)
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            
            # [OPTIMIZED] More efficient than zero_grad()
            optimizer.zero_grad(set_to_none=True)
        else:
            # Standard training (FP32)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        # Monitor gradients
        grad_stats = monitor_gradients(model)
        grad_norms.append(grad_stats['total_norm'])
        
        # Critical: Check for NaN/Inf in gradients
        if grad_stats['nan_count'] > 0:
            raise ValueError(f"🚨 TRAINING COLLAPSED: NaN gradients detected at batch {batch_idx}")
        if grad_stats['inf_count'] > 0:
            raise ValueError(f"🚨 TRAINING COLLAPSED: Inf gradients detected at batch {batch_idx}")
        
        # Check for loss spikes
        if loss_monitor.is_spiking():
            print(f"\n⚠️  WARNING: Loss spike detected at batch {batch_idx}")
            print(f"   Recent losses: {[f'{l:.2f}' for l in loss_monitor.losses[-5:]]}")
        
        # Check for loss divergence
        if loss_monitor.is_diverging():
            print(f"\n⚠️  WARNING: Loss diverging at batch {batch_idx}")
            print(f"   Recent losses: {[f'{l:.2f}' for l in loss_monitor.losses[-5:]]}")
        
        # Print detailed statistics every 100 batches
        if (batch_idx + 1) % 100 == 0:
            perplexity = calculate_perplexity(loss.item())
            print(f"\n[Batch {batch_idx + 1}] Loss: {loss.item():.4f} | PPL: {perplexity:.2f} | Grad: {grad_stats['total_norm']:.4f}")
            
            # Check activation anomalies every 100 batches
            warnings = activation_monitor.check_anomalies()
            if warnings:
                print("Activation Warnings:")
                for warning in warnings:
                    print(f"  {warning}")
        
        total_loss += loss.item()
        batch_idx += 1
    
    # Print epoch summary
    avg_loss = total_loss / len(train_iter)
    avg_perplexity = calculate_perplexity(avg_loss)
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
    min_loss = min(batch_losses) if batch_losses else 0
    max_loss = max(batch_losses) if batch_losses else 0
    
    print(f"\n[Epoch {epoch + 1} Summary]")
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {avg_perplexity:.2f}")
    print(f"  Avg Gradient Norm: {avg_grad_norm:.4f}")
    print(f"  Loss Range: [{min_loss:.4f}, {max_loss:.4f}]")
    
    return avg_loss


def adjust_learning_rate(optimizer, epoch, decay_start):
    """
    [Paper Sec 4.1] Halve learning rate every epoch after decay_start epoch.
    
    Paper: "after N epochs, we begin to halve the learning rate every epoch"
    - epoch is 0-indexed here
    - If decay_start=5, decay starts from epoch 6 (after epochs 0-5 are done)
    
    Args:
        optimizer: Optimizer instance
        epoch: Current epoch number (0-indexed)
        decay_start: Epoch after which to start decaying
    """
    if epoch > decay_start:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
            print(f"Decaying learning rate to {param_group['lr']}")


def main():
    args = parse_args()
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if cuda else 'cpu')

    # ===== Reproducibility =====
    torch.manual_seed(42)
    if cuda:
        torch.cuda.manual_seed_all(42)

    print("=" * 70)
    print("Using %s for training" % ('GPU' if cuda else 'CPU'))
    
    # [OPTIMIZED] AMP setup
    use_amp = bool(getattr(args, 'amp', False))
    scaler = GradScaler(enabled=use_amp) if cuda else None
    
    if use_amp:
        print("⚡ Automatic Mixed Precision (AMP) enabled")
        print("   Expected speedup: 1.5-2x")
        print("   Note: May affect bit-level reproducibility")
    
    print("=" * 70)
    
    # ===== Dataset Loading =====
    print('Loading dataset...', end='', flush=True)
    metadata, vocab, train_iter, val_iter, test_iter = dataset_factory(args, device)
    print('Done.')

    print('Saving vocab and args...', end='', flush=True)
    save_vocab(vocab, args.save_path + os.path.sep + 'vocab')
    save_object(args, args.save_path + os.path.sep + 'args')
    print('Done')

    # ===== Model Initialization =====
    model = train_model_factory(args, metadata)
    model = model.to(device)
    
    if cuda and args.multi_gpu:
        model = nn.DataParallel(model, dim=1)
    
    # [OPTIMIZED] Print model info
    print("\n" + "=" * 70)
    print("Model Architecture:")
    print("=" * 70)
    print(model)
    print("=" * 70)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print("=" * 70)

    # ===== Optimizer Setup =====
    # [Paper] Optimizer: SGD with lr=1.0
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    
    print(f"\nOptimizer: SGD")
    print(f"Initial learning rate: {args.learning_rate}")
    print(f"Learning rate decay starts after epoch: {args.lr_decay_start}")
    print(f"Gradient clipping: {args.gradient_clip}")
    print("=" * 70 + "\n")

    # ===== Training Loop =====
    try:
        best_val_loss = None
        
        for epoch in range(args.max_epochs):
            start = datetime.now()
            
            # [Paper] LR Scheduling: Halve after specified epoch
            if epoch > args.lr_decay_start:
                adjust_learning_rate(optimizer, epoch, args.lr_decay_start)

            # Train for one epoch with monitoring
            train_loss = train(
                model=model,
                optimizer=optimizer,
                train_iter=train_iter,
                metadata=metadata,
                grad_clip=args.gradient_clip,
                reverse_src=args.reverse,
                use_amp=use_amp,
                scaler=scaler,
                epoch=epoch
            )
            
            # Evaluate on validation set
            val_loss = evaluate(
                model=model,
                val_iter=val_iter,
                metadata=metadata,
                reverse_src=args.reverse
            )
            
            # Print epoch results
            elapsed = datetime.now() - start
            print(f"[Epoch {epoch + 1:2d}/{args.max_epochs}] "
                  f"train_loss: {train_loss:.4f} - "
                  f"val_loss: {val_loss:.4f} - "
                  f"time: {elapsed}", end='')

            # Save model
            if args.save_every_epoch or not best_val_loss or val_loss < best_val_loss:
                print(' (Saving model...', end='', flush=True)
                save_model(args.save_path, model, epoch + 1, train_loss, val_loss)
                print('Done)', end='')
                best_val_loss = val_loss
            
            print()  # New line
            
            # Sample generation check after evaluation
            sample_generation(model, val_iter, metadata, reverse_src=args.reverse, num_samples=3)
            
    except ValueError as e:
        # Training collapse detected
        print(f"\n{'='*70}")
        print(f"TRAINING STOPPED DUE TO COLLAPSE")
        print(f"{'='*70}")
        print(f"Error: {e}")
        print(f"{'='*70}")
        raise
    except (KeyboardInterrupt, BrokenPipeError):
        print('\n[Ctrl-C] Training stopped by user.')

    # ===== Final Evaluation =====
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)
    
    test_loss = evaluate(
        model=model,
        val_iter=test_iter,
        metadata=metadata,
        reverse_src=args.reverse
    )
    
    print(f"Test loss: {test_loss:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
