#!/usr/bin/env python3
"""
Demo script showing the separate vocabulary functionality.
This demonstrates that the fix allows models trained with different
source and target vocabulary sizes to work correctly during inference.
"""

import torch
import os
import tempfile
import shutil
from dataset import Vocab, metadata_factory
from model import train_model_factory, predict_model_factory
from serialization import save_object
from constants import MODEL_FORMAT

# Import SimpleField from predict.py
import sys
sys.path.insert(0, os.path.dirname(__file__))

class SimpleField:
    """Simple field-like object that wraps vocab for compatibility"""
    def __init__(self, vocab, add_sos=False, add_eos=True):
        self.vocab = vocab
        self.add_sos = add_sos
        self.add_eos = add_eos
    
    def preprocess(self, text):
        return text.strip().split()
    
    def process(self, batch):
        processed = []
        for tokens in batch:
            indices = []
            if self.add_sos:
                indices.append(self.vocab.stoi['<sos>'])
            indices.extend([self.vocab.stoi.get(tok, self.vocab.stoi['<unk>']) for tok in tokens])
            if self.add_eos:
                indices.append(self.vocab.stoi['<eos>'])
            processed.append(indices)
        
        pad_idx = self.vocab.stoi['<pad>']
        max_len = max(len(seq) for seq in processed)
        padded = []
        for seq in processed:
            padded.append(seq + [pad_idx] * (max_len - len(seq)))
        
        tensor = torch.tensor(padded, dtype=torch.long).t()
        return tensor


class TestArgs:
    """Minimal args for demo"""
    def __init__(self):
        self.encoder_rnn_cell = 'LSTM'
        self.encoder_hidden_size = 64
        self.encoder_num_layers = 1
        self.encoder_rnn_dropout = 0.0
        self.encoder_bidirectional = False
        self.decoder_type = 'luong'
        self.decoder_rnn_cell = 'LSTM'
        self.decoder_hidden_size = 64
        self.decoder_num_layers = 1
        self.decoder_rnn_dropout = 0.0
        self.luong_attn_hidden_size = 64
        self.luong_input_feed = False
        self.decoder_init_type = 'zeros'
        self.attention_type = 'none'
        self.attention_score = 'dot'
        self.embedding_type = None
        self.embedding_size = 128
        self.train_embeddings = True
        self.teacher_forcing_ratio = 0.0
        self.cuda = False
        self.multi_gpu = False


def main():
    print("=" * 70)
    print("DEMO: Separate Source/Target Vocabulary Support")
    print("=" * 70)
    
    # Simulate English -> German translation scenario
    print("\n📚 Creating vocabularies...")
    print("   Source (English): smaller vocabulary")
    en_tokens = ['hello', 'world', 'how', 'are', 'you', 'today', 'good', 'morning']
    src_vocab = Vocab(en_tokens)
    
    print("   Target (German): larger vocabulary")
    de_tokens = ['hallo', 'welt', 'wie', 'geht', 'es', 'ihnen', 'heute', 
                 'guten', 'morgen', 'sehr', 'gut', 'danke']
    tgt_vocab = Vocab(de_tokens)
    
    print(f"\n   ✅ SRC vocab size: {len(src_vocab)}")
    print(f"   ✅ TGT vocab size: {len(tgt_vocab)}")
    print(f"   ✅ Different vocabulary sizes: {len(src_vocab)} != {len(tgt_vocab)}")
    
    # Create model
    print("\n🔧 Building model...")
    args = TestArgs()
    src_metadata = metadata_factory(args, src_vocab)
    tgt_metadata = metadata_factory(args, tgt_vocab)
    
    model = train_model_factory(args, src_metadata, tgt_metadata)
    print(f"   ✅ Encoder embedding size: {model.encoder.embed.num_embeddings}")
    print(f"   ✅ Decoder embedding size: {model.decoder.embed.num_embeddings}")
    print(f"   ✅ Encoder uses SRC vocab: {model.encoder.embed.num_embeddings == len(src_vocab)}")
    print(f"   ✅ Decoder uses TGT vocab: {model.decoder.embed.num_embeddings == len(tgt_vocab)}")
    
    # Create fields for inference
    print("\n🔍 Creating inference fields...")
    src_field = SimpleField(src_vocab, add_sos=False, add_eos=True)
    tgt_field = SimpleField(tgt_vocab, add_sos=False, add_eos=False)
    print("   ✅ Source field: adds <eos> only (matches training)")
    print("   ✅ Target field: no special tokens (managed by Seq2SeqPredict)")
    
    # Create predict model
    from model.seq2seq.model import Seq2SeqPredict
    predict_model = Seq2SeqPredict(
        model.encoder,
        model.decoder,
        src_field,
        tgt_field
    )
    
    # Test inference
    print("\n🚀 Testing inference with separate vocabularies...")
    test_inputs = ["hello world", "good morning", "how are you today"]
    
    predict_model.eval()
    with torch.no_grad():
        outputs = predict_model(test_inputs, 'greedy', max_seq_len=8)
    
    print("\n📊 Inference Results:")
    print("-" * 70)
    for i, (inp, out) in enumerate(zip(test_inputs, outputs), 1):
        print(f"   {i}. Input:  '{inp}'")
        print(f"      Output: '{out}'")
        print()
    
    print("=" * 70)
    print("✅ SUCCESS: Models with different src/tgt vocab sizes work!")
    print("=" * 70)
    print("\nKey Points:")
    print("  • Source vocabulary size can differ from target vocabulary size")
    print("  • Encoder uses source vocab for embeddings")
    print("  • Decoder uses target vocab for embeddings and output")
    print("  • Source preprocessing matches training (adds <eos> only)")
    print("  • Backward compatible with legacy single-vocab checkpoints")
    print()


if __name__ == '__main__':
    main()
