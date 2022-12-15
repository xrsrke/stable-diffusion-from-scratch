import torch
from foundation.transformer.efficient_attention import ScaleDotProductAttention, MultiHeadAttention


def test_scaled_dot_product_attention(attention_parameters, attention_batch):
    batch_size, n_words = attention_parameters['batch_size'], attention_parameters['n_words']
    d_model = attention_parameters['d_model']
    n_heads, d_head = attention_parameters['n_heads'], attention_parameters['d_head']

    q_batch, k_batch, v_batch = attention_batch

    attention = ScaleDotProductAttention(d_head=d_head)
    output, attention_weights = attention(
        q_batch=q_batch,
        k_batch=k_batch,
        v_batch=v_batch
    )

    assert output.shape == (batch_size, n_heads, n_words, d_head)
    assert attention_weights.shape == (batch_size, n_heads, n_words, n_words)
    # assert attention_weights[0][1].sum().item() == n_words
    assert attention_weights[0][1][0].sum().item() == 1.

# def test_multi_head_attention():
#     attention = ScaleDotProductAttention(d_head=None)
#     mha = MultiHeadAttention(
#         attention=attention,
#     )