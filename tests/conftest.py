import torch
import pytest

@pytest.fixture
def attention_parameters():
    batch_size = 8
    n_words = 7
    d_model = 10
    n_heads = 2
    d_head = d_model // n_heads

    return {
        'batch_size': batch_size,
        'n_words': n_words,
        'd_model': d_model,
        'n_heads': n_heads,
        'd_head': d_head
    }

@pytest.fixture
def attention_batch(attention_parameters):
    parameters = attention_parameters

    batch_size = parameters['batch_size']
    n_heads = parameters['n_heads']
    n_words = parameters['n_words']
    d_head = parameters['d_head']

    q_batch = torch.randn(batch_size, n_heads, n_words, d_head)
    k_batch = torch.randn(batch_size, n_heads, n_words, d_head)
    v_batch = torch.randn(batch_size, n_heads, n_words, d_head)

    return q_batch, k_batch, v_batch