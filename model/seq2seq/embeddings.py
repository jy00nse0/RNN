import torch.nn as nn

def embeddings_factory(args, metadata):
    embed = nn.Embedding(num_embeddings=metadata.vocab_size, embedding_dim=args.embedding_size,
                         padding_idx=metadata.padding_idx, _weight=metadata.vectors)
    
    # [Fixed] Typos fix: require_grads -> requires_grad
    # This ensures that embedding freezing works as intended (e.g. for pre-trained vectors)
    embed.weight.requires_grad = args.train_embeddings
    
    return embed
