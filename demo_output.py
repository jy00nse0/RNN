#!/usr/bin/env python3
"""
Demo script to show the enhanced training output format.
This simulates what users will see during training.
"""

import sys
import time
sys.path.insert(0, '/home/runner/work/RNN/RNN')

from train import calculate_perplexity, log_batch_statistics

def simulate_epoch():
    """Simulate one training epoch with the new logging"""
    
    print("\n" + "=" * 70)
    print("Epoch 1/10")
    print("=" * 70)
    
    # Simulate batch logging (every 100 batches)
    batches = [0, 100, 200, 300]
    losses = [8.5234, 7.2341, 6.8123, 6.4567]
    grad_norms = [2.3451, 1.8234, 1.6789, 1.5234]
    
    for batch_idx, loss, grad_norm in zip(batches, losses, grad_norms):
        log_batch_statistics(batch_idx, 350, loss, grad_norm, 1.0)
        time.sleep(0.1)  # Small delay for readability
    
    print()
    
    # Validation metrics
    val_loss = 6.8234
    val_ppl = calculate_perplexity(val_loss)
    min_loss = 6.1234
    max_loss = 8.9012
    
    print(f"\n  📊 Validation Metrics:")
    print(f"     Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")
    print(f"     Min Loss: {min_loss:.4f} | Max Loss: {max_loss:.4f}")
    
    # Sample translations
    print("\n  🔍 Sample Translations:")
    
    samples = [
        {
            'source': '<sos> how are you today ? <eos>',
            'target': '<sos> wie geht es dir heute ? <eos>',
            'prediction': '<sos> wie sind sie heute ? <eos>'
        },
        {
            'source': '<sos> thank you very much <eos>',
            'target': '<sos> vielen dank <eos>',
            'prediction': '<sos> danke sehr <eos>'
        }
    ]
    
    for i, sample in enumerate(samples, 1):
        print(f"\n  Example {i}:")
        print(f"    SRC: {sample['source']}")
        print(f"    TGT: {sample['target']}")
        print(f"    PRD: {sample['prediction']}")
    
    # Epoch summary
    train_loss = 7.5234
    train_ppl = calculate_perplexity(train_loss)
    avg_grad_norm = 1.9876
    
    print("\n" + "=" * 70)
    print(f"[Epoch  1/10] Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
    print(f"  Avg Grad Norm: {avg_grad_norm:.4f}")
    print(f"  Time: 0:45:23")
    print("=" * 70)
    
    # Model saving
    print(f"\n💾 Saving model (val_loss improved: inf → {val_loss:.4f})")
    print()

def main():
    """Run the demonstration"""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Enhanced Training Output")
    print("=" * 70)
    print("\nThis shows what users will see during training with the new")
    print("debugging and monitoring features:")
    print("  ✓ Batch-level progress logging (every 100 batches)")
    print("  ✓ Perplexity calculations")
    print("  ✓ Validation metrics with min/max")
    print("  ✓ Sample translations to visualize learning")
    print("  ✓ Gradient norm tracking")
    print("  ✓ Clear epoch summaries")
    print("  ✓ Improved model saving feedback")
    
    simulate_epoch()
    
    print("\n" + "=" * 70)
    print("Additional Features (not shown in this demo):")
    print("=" * 70)
    print("  ✓ NaN/Inf detection with warnings")
    print("  ✓ Metrics saved to 'training_metrics.jsonl' for analysis")
    print("  ✓ All existing functionality preserved (backward compatible)")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
