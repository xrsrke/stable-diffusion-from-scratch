import torch
from foundation.transformer.encoder import (
    ResidualLayerNorm,
    PostionWiseFeedForward,
    EncoderLayer,
    Encoder
)

def test_residual_layer_norm():
    d_model = 10
    dropout = 0.2

    layer = ResidualLayerNorm(d_model, dropout)
    batch_size = 5
    seq_len = 4
    x = torch.randn(batch_size, seq_len, d_model)

    output = layer(x, x)

    assert output.shape == (batch_size, seq_len, d_model)

def test_postion_wise_feed_forward():
    d_model = 10
    d_ff = 16
    dropout = 0.3

    batch_size = 5
    seq_len = 4
    tokens = torch.randn(batch_size, seq_len, d_model)

    layer = PostionWiseFeedForward(
        d_model = d_model,
        d_ff = d_ff,
        dropout = dropout
    )
    output = layer(tokens)

    assert output.shape == (batch_size, seq_len, d_model)

def test_encoder_layer():
    d_model = 10
    n_heads = 2
    d_ff = 16

    layer = EncoderLayer(
        d_model, n_heads, d_ff, dropout = 0.3
    )

    batch_size = 5
    seq_len = 4
    x = torch.randn(batch_size, seq_len, d_model)

    output, attention_weights = layer(x, mask=None)

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, n_heads, seq_len, seq_len)

def test_encoder():

    batch_size = 6
    seq_len = 5
    d_model = 10
    n_heads = 2

    tokens = torch.arange(0, 30).reshape(batch_size, seq_len)
    encoder = Encoder(
        d_model=d_model,
        n_heads=n_heads,
        n_layers = 3,
        d_ff=16,
        dropout=0.2
    )

    encoding, encoder_attention_weights = encoder(tokens)

    assert encoding.shape == (batch_size, seq_len, d_model)
    assert encoder_attention_weights.shape == (batch_size, n_heads, seq_len, seq_len)