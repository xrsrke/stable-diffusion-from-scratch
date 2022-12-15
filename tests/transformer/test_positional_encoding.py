from foundation.transformer.positional_encoding import PositionalEncoding

def test_positional_encoding():
    d_model = 10
    max_seq_len = 1000
    dropout = 0.3

    positional_encoding = PositionalEncoding(
        d_model = d_model,
        max_seq_len = max_seq_len,
        dropout = dropout,
    )

