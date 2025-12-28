#!/usr/bin/env python3
"""
Example: Generate loss graph from existing training metrics

This script demonstrates how to manually generate loss graphs from
training_metrics.jsonl files that were created during training.

Usage:
    python generate_loss_graph.py --metrics-file /path/to/training_metrics.jsonl --output loss_graph.png
    
    Or with default paths:
    python generate_loss_graph.py
"""

import argparse
import os
import sys
from util import load_training_metrics, plot_loss_graph


def main():
    parser = argparse.ArgumentParser(
        description='Generate loss graph from training metrics file'
    )
    parser.add_argument(
        '--metrics-file',
        type=str,
        default='training_metrics.jsonl',
        help='Path to training_metrics.jsonl file (default: training_metrics.jsonl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='loss_graph.png',
        help='Output path for the loss graph (default: loss_graph.png)'
    )
    
    args = parser.parse_args()
    
    # Check if metrics file exists
    if not os.path.exists(args.metrics_file):
        print(f"Error: Metrics file not found: {args.metrics_file}")
        print("\nPlease provide a valid path to training_metrics.jsonl")
        print("This file is created automatically during training in the save_path directory.")
        return 1
    
    try:
        # Load metrics
        print(f"Loading metrics from: {args.metrics_file}")
        train_losses, val_losses = load_training_metrics(args.metrics_file)
        
        if not train_losses or not val_losses:
            print("Error: No loss data found in metrics file")
            return 1
        
        print(f"Found {len(train_losses)} epochs of training data")
        print(f"Train loss range: [{min(train_losses):.4f}, {max(train_losses):.4f}]")
        print(f"Val loss range: [{min(val_losses):.4f}, {max(val_losses):.4f}]")
        
        # Generate graph
        print(f"\nGenerating loss graph...")
        plot_loss_graph(train_losses, val_losses, args.output)
        
        # Verify file was created
        if os.path.exists(args.output):
            file_size = os.path.getsize(args.output)
            print(f"✓ Success! Graph saved to: {args.output}")
            print(f"  File size: {file_size:,} bytes")
            return 0
        else:
            print("✗ Error: Graph file was not created")
            return 1
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
