#!/usr/bin/env python3
"""
Create minimal mock data for testing the RNN training pipeline.
This creates small synthetic datasets that mimic the structure of WMT14 data.
"""

import os
import random

def create_mock_data():
    """Create mock WMT14 data with minimal size for testing"""
    
    # Create directory structure
    base_dir = "data/wmt14_vocab50k/base"
    os.makedirs(base_dir, exist_ok=True)
    
    # Simple word lists for generating sentences
    en_words = ["the", "a", "an", "this", "that", "is", "was", "are", "were",
                "I", "you", "he", "she", "it", "we", "they",
                "go", "went", "come", "came", "see", "saw", "do", "did",
                "good", "bad", "big", "small", "new", "old",
                "man", "woman", "child", "dog", "cat", "house", "car", "book",
                "to", "from", "in", "on", "at", "with", "for", "by",
                "hello", "goodbye", "please", "thank", "yes", "no", "maybe"]
    
    de_words = ["der", "die", "das", "ein", "eine", "dieser", "ist", "war", "sind", "waren",
                "ich", "du", "er", "sie", "es", "wir", "ihr",
                "gehen", "ging", "kommen", "kam", "sehen", "sah", "tun", "tat",
                "gut", "schlecht", "gross", "klein", "neu", "alt",
                "Mann", "Frau", "Kind", "Hund", "Katze", "Haus", "Auto", "Buch",
                "zu", "von", "in", "auf", "an", "mit", "fuer", "durch",
                "hallo", "tschuess", "bitte", "danke", "ja", "nein", "vielleicht"]
    
    def generate_sentence(words, min_len=3, max_len=15):
        """Generate a random sentence from word list"""
        length = random.randint(min_len, max_len)
        return " ".join(random.choices(words, k=length))
    
    # Generate datasets with different sizes
    datasets = {
        "train": 150000,  # Reduced from original for faster testing
        "valid": 3000,
        "test": 3000
    }
    
    for split, num_lines in datasets.items():
        en_file = os.path.join(base_dir, f"{split}.en")
        de_file = os.path.join(base_dir, f"{split}.de")
        
        print(f"Creating {split} set with {num_lines} lines...")
        
        with open(en_file, "w", encoding="utf-8") as fen, \
             open(de_file, "w", encoding="utf-8") as fde:
            
            for _ in range(num_lines):
                en_sent = generate_sentence(en_words)
                de_sent = generate_sentence(de_words)
                
                fen.write(en_sent + "\n")
                fde.write(de_sent + "\n")
    
    print(f"Mock data created in {base_dir}/")
    print("Files created:")
    print("  - train.en, train.de")
    print("  - valid.en, valid.de")
    print("  - test.en, test.de")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    create_mock_data()
    print("\nDone!")
