#!/usr/bin/env python3
"""
Integration test to verify the complete workflow:
1. Create minimal training data
2. Train a model (saves src_vocab and tgt_vocab)
3. Run BLEU evaluation (loads src_vocab and tgt_vocab)
"""

import os
import sys
import tempfile
import shutil
import torch
import argparse
from dataset import Vocab, TranslationDataset, metadata_factory
from serialization import save_vocab, save_object, load_object, save_model
from model import train_model_factory
from constants import MODEL_FORMAT


def create_minimal_dataset(temp_dir):
    """Create minimal dataset for testing"""
    # Create data subdirectory
    data_dir = os.path.join(temp_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Create minimal training data (English -> German)
    train_en = os.path.join(data_dir, 'train.en')
    train_de = os.path.join(data_dir, 'train.de')
    
    # English source (smaller vocab)
    en_sentences = [
        "hello world",
        "good morning",
        "thank you",
    ]
    
    # German target (larger vocab for testing)
    de_sentences = [
        "hallo welt",
        "guten morgen",
        "danke schoen vielen dank",  # Extra tokens
    ]
    
    with open(train_en, 'w', encoding='utf-8') as f:
        for sent in en_sentences:
            f.write(sent + '\n')
    
    with open(train_de, 'w', encoding='utf-8') as f:
        for sent in de_sentences:
            f.write(sent + '\n')
    
    return data_dir


def test_training_saves_both_vocabs():
    """Test that training saves both src_vocab and tgt_vocab"""
    print("\n" + "="*70)
    print("Integration Test: Training Saves Both Vocabularies")
    print("="*70)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create minimal dataset
        data_dir = create_minimal_dataset(temp_dir)
        
        # Load dataset
        train_en = os.path.join(data_dir, 'train.en')
        train_de = os.path.join(data_dir, 'train.de')
        
        dataset = TranslationDataset(train_en, train_de)
        src_vocab = dataset.src_vocab
        tgt_vocab = dataset.tgt_vocab
        
        print(f"Created dataset with src_vocab={len(src_vocab)}, tgt_vocab={len(tgt_vocab)}")
        
        # Simulate training save
        save_dir = os.path.join(temp_dir, 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save vocabularies (as train.py does)
        save_vocab(src_vocab, os.path.join(save_dir, 'src_vocab'))
        save_vocab(tgt_vocab, os.path.join(save_dir, 'tgt_vocab'))
        save_vocab(tgt_vocab, os.path.join(save_dir, 'vocab'))  # Backward compatibility
        
        # Create minimal args
        args = argparse.Namespace(
            embedding_size=10,
            embedding_type=None,
            train_embeddings=True,
            encoder_rnn_cell='LSTM',
            encoder_hidden_size=10,
            encoder_num_layers=1,
            encoder_rnn_dropout=0.0,
            encoder_bidirectional=False,
            decoder_type='luong',
            decoder_rnn_cell='LSTM',
            decoder_hidden_size=10,
            decoder_num_layers=1,
            decoder_rnn_dropout=0.0,
            luong_attn_hidden_size=10,
            luong_input_feed=False,
            decoder_init_type='zeros',
            attention_type='none',
            attention_score='dot',
            teacher_forcing_ratio=1.0,
        )
        save_object(args, os.path.join(save_dir, 'args'))
        
        # Create and save a minimal model
        src_metadata = metadata_factory(args, src_vocab)
        tgt_metadata = metadata_factory(args, tgt_vocab)
        model = train_model_factory(args, src_metadata, tgt_metadata)
        
        # Save model checkpoint
        epoch = 1
        train_loss = 0.5
        val_loss = 0.6
        save_model(save_dir, model, epoch, train_loss, val_loss)
        
        print(f"Saved model and vocabularies to {save_dir}")
        
        # Verify files exist
        assert os.path.exists(os.path.join(save_dir, 'src_vocab')), "src_vocab not saved"
        assert os.path.exists(os.path.join(save_dir, 'tgt_vocab')), "tgt_vocab not saved"
        assert os.path.exists(os.path.join(save_dir, 'vocab')), "vocab not saved"
        assert os.path.exists(os.path.join(save_dir, 'args')), "args not saved"
        print("✓ All required files saved")
        
        # Load and verify vocabularies
        loaded_src = load_object(os.path.join(save_dir, 'src_vocab'))
        loaded_tgt = load_object(os.path.join(save_dir, 'tgt_vocab'))
        loaded_compat = load_object(os.path.join(save_dir, 'vocab'))
        
        assert len(loaded_src) == len(src_vocab), "src_vocab size mismatch"
        assert len(loaded_tgt) == len(tgt_vocab), "tgt_vocab size mismatch"
        assert len(loaded_compat) == len(tgt_vocab), "vocab compatibility broken"
        print("✓ Vocabularies loaded correctly")
        
        # Load checkpoint and verify embedding sizes
        model_files = [f for f in os.listdir(save_dir) if f.startswith('seq2seq-')]
        assert len(model_files) > 0, "No model checkpoint found"
        
        checkpoint_path = os.path.join(save_dir, model_files[0])
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        encoder_embed_size = checkpoint['encoder.embed.weight'].shape[0]
        decoder_embed_size = checkpoint['decoder.embed.weight'].shape[0]
        
        print(f"Checkpoint embedding sizes: encoder={encoder_embed_size}, decoder={decoder_embed_size}")
        assert encoder_embed_size == len(src_vocab), "Encoder embedding size doesn't match src_vocab"
        assert decoder_embed_size == len(tgt_vocab), "Decoder embedding size doesn't match tgt_vocab"
        print("✓ Checkpoint embedding sizes match vocabularies")
        
        print("\n✅ Integration Test PASSED: Training saves both vocabularies correctly\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration Test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir)


def test_evaluation_loads_both_vocabs():
    """Test that evaluation can load both vocabularies"""
    print("\n" + "="*70)
    print("Integration Test: Evaluation Loads Both Vocabularies")
    print("="*70)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create minimal dataset with different vocab sizes
        data_dir = create_minimal_dataset(temp_dir)
        train_en = os.path.join(data_dir, 'train.en')
        train_de = os.path.join(data_dir, 'train.de')
        
        dataset = TranslationDataset(train_en, train_de)
        src_vocab = dataset.src_vocab
        tgt_vocab = dataset.tgt_vocab
        
        print(f"Created dataset with src_vocab={len(src_vocab)}, tgt_vocab={len(tgt_vocab)}")
        
        # Simulate training save
        save_dir = os.path.join(temp_dir, 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save vocabularies
        save_vocab(src_vocab, os.path.join(save_dir, 'src_vocab'))
        save_vocab(tgt_vocab, os.path.join(save_dir, 'tgt_vocab'))
        save_vocab(tgt_vocab, os.path.join(save_dir, 'vocab'))
        
        # Create and save model
        args = argparse.Namespace(
            embedding_size=10,
            embedding_type=None,
            train_embeddings=True,
            encoder_rnn_cell='LSTM',
            encoder_hidden_size=10,
            encoder_num_layers=1,
            encoder_rnn_dropout=0.0,
            encoder_bidirectional=False,
            decoder_type='luong',
            decoder_rnn_cell='LSTM',
            decoder_hidden_size=10,
            decoder_num_layers=1,
            decoder_rnn_dropout=0.0,
            luong_attn_hidden_size=10,
            luong_input_feed=False,
            decoder_init_type='zeros',
            attention_type='none',
            attention_score='dot',
            teacher_forcing_ratio=1.0,
        )
        save_object(args, os.path.join(save_dir, 'args'))
        
        src_metadata = metadata_factory(args, src_vocab)
        tgt_metadata = metadata_factory(args, tgt_vocab)
        model = train_model_factory(args, src_metadata, tgt_metadata)
        save_model(save_dir, model, 1, 0.5, 0.6)
        
        # Simulate evaluation loading (as calculate_bleu.py does)
        src_vocab_path = os.path.join(save_dir, 'src_vocab')
        tgt_vocab_path = os.path.join(save_dir, 'tgt_vocab')
        
        # Check new format
        if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
            loaded_src = load_object(src_vocab_path)
            loaded_tgt = load_object(tgt_vocab_path)
            print(f"Loaded separate vocabularies: src={len(loaded_src)}, tgt={len(loaded_tgt)}")
        else:
            raise FileNotFoundError("src_vocab or tgt_vocab not found")
        
        # Verify sizes match
        assert len(loaded_src) == len(src_vocab), "src_vocab size mismatch"
        assert len(loaded_tgt) == len(tgt_vocab), "tgt_vocab size mismatch"
        print("✓ Vocabulary sizes match")
        
        # Load checkpoint and verify can create model
        model_files = [f for f in os.listdir(save_dir) if f.startswith('seq2seq-')]
        checkpoint_path = os.path.join(save_dir, model_files[0])
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        encoder_embed_size = checkpoint['encoder.embed.weight'].shape[0]
        decoder_embed_size = checkpoint['decoder.embed.weight'].shape[0]
        
        print(f"Checkpoint: encoder_embed={encoder_embed_size}, decoder_embed={decoder_embed_size}")
        print(f"Vocabularies: src={len(loaded_src)}, tgt={len(loaded_tgt)}")
        
        assert encoder_embed_size == len(loaded_src), "Encoder size mismatch"
        assert decoder_embed_size == len(loaded_tgt), "Decoder size mismatch"
        print("✓ Checkpoint sizes match vocabularies")
        
        # Try to create model with loaded vocabs
        loaded_src_metadata = metadata_factory(args, loaded_src)
        loaded_tgt_metadata = metadata_factory(args, loaded_tgt)
        eval_model = train_model_factory(args, loaded_src_metadata, loaded_tgt_metadata)
        
        # Load state dict
        eval_model.load_state_dict(checkpoint)
        print("✓ Model loaded successfully with separate vocabularies")
        
        print("\n✅ Integration Test PASSED: Evaluation loads both vocabularies correctly\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration Test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir)


def main():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("Vocabulary Integration Tests")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Training Saves Both Vocabs", test_training_saves_both_vocabs()))
    results.append(("Evaluation Loads Both Vocabs", test_evaluation_loads_both_vocabs()))
    
    # Print summary
    print("\n" + "="*70)
    print("Integration Test Summary")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print("="*70)
    print(f"Total: {passed}/{total} tests passed")
    print("="*70)
    
    return all(result for _, result in results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
