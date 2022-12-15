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

    torch.testing.assert_close(attention_weights[0][1].sum().item(), float(n_words))
    torch.testing.assert_close(attention_weights[0][1][0].sum().item(), float(1))

def test_multi_head_attention(attention_parameters):
    batch_size = attention_parameters['batch_size']
    d_model = attention_parameters['d_model']
    n_heads = attention_parameters['n_heads']
    n_words = attention_parameters['n_words']

    mha = MultiHeadAttention(
        d_model=d_model,
        n_head=n_heads
    )

    q = torch.randn(batch_size, n_words, d_model)
    k = torch.randn(batch_size, n_words, d_model)
    v = torch.randn(batch_size, n_words, d_model)

    output, attention_weights = mha(
        pre_q=q,
        pre_k=k,
        pre_v=v
    )

    assert output.shape == (batch_size, n_words, d_model)
    assert attention_weights.shape == (batch_size, n_heads, n_words, n_words)