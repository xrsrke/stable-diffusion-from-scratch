import torch
from foundation.transformer.embedding import TextEmbedding

def test_text_embedding():
    vocab_size = 100
    padding_idx = 0
    d_model = 10

    tokens = torch.tensor([
        [0, 1, 2, 3, 4, 0],
        [6, 7, 8, 9, 10, 11]
    ])

    text_embedding = TextEmbedding(
        vocab_size = vocab_size,
        d_model = d_model,
        padding_idx = padding_idx
    )

    embeddings = text_embedding(tokens)

    assert embeddings.shape == (2, 6, 10)