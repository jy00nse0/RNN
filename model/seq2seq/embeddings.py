import torch.nn as nn
import torch


def embeddings_factory(args, metadata):
    """
    Build embedding layer.
    - If metadata.vectors is provided (pretrained), use it.
    - Otherwise, initialize uniformly in [-0.1, 0.1] per Luong et al. (2015).
    """
    if metadata.vectors is not None:
        embed = nn.Embedding(
            num_embeddings=metadata.vocab_size,
            embedding_dim=args.embedding_size,
            padding_idx=metadata.padding_idx,
            _weight=metadata.vectors
        )
    else:
        embed = nn.Embedding(
            num_embeddings=metadata.vocab_size,
            embedding_dim=args.embedding_size,
            padding_idx=metadata.padding_idx
        )
        # Paper: Uniform initialization in [-0.1, 0.1]
        nn.init.uniform_(embed.weight.data, -0.1, 0.1)

    # Ensure embedding training flag
    embed.weight.requires_grad = args.train_embeddings

    return embed
