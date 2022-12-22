import torch

from foundation.transformer.all import Transformer


def test_transformer():

    batch_size = 2
    src = torch.tensor([
        [1, 5, 6, 4, 3, 9, 5, 2, 0],
        [1, 8, 7, 3, 4, 5, 6, 7, 2]
    ])

    trg_seq_len = 7
    trg = torch.tensor([
        [1, 7, 4, 3, 5, 9, 2, 0],
        [1, 5, 6, 2, 4, 7, 6, 2]
    ])

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_len = 10
    trg_vocab_len = 10

    d_model=10
    d_ff = 16
    n_layers=3
    n_heads = 2


    model = Transformer(
        src_vocab_len, trg_vocab_len, d_model=d_model, d_ff=d_ff,
        n_layers=n_layers, n_heads=n_heads, src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx,
    )

    output = model(src, trg[:, :-1])

    assert output.shape == (batch_size, trg_seq_len, d_model)