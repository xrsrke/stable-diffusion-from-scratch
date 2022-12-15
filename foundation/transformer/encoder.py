# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/transformer/07_transformer.encoder.ipynb.

# %% auto 0
__all__ = ['ResidualLayerNorm', 'PostionWiseFeedForward', 'EncoderLayer', 'Encoder']

# %% ../../nbs/transformer/07_transformer.encoder.ipynb 4
from typing import Callable

import torch
from torch import nn

from .attention import MultiHeadAttention
from .embedding import PositionalEncoding

# %% ../../nbs/transformer/07_transformer.encoder.ipynb 5
class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model: int, dropout=0.3):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor):
        return self.layer_norm(self.dropout(x + residual))

# %% ../../nbs/transformer/07_transformer.encoder.ipynb 6
class PostionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout=0.3):
        super().__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor):
        # shape(x) = [B x seq_len x D]

        output = self.feed_forward(x)
        # shape(output) = [B x seq_len x D]

        return output

# %% ../../nbs/transformer/07_transformer.encoder.ipynb 7
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.3):
        super().__init__()
        self.norm_1 = ResidualLayerNorm(d_model, dropout)
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm_2 = ResidualLayerNorm(d_model, dropout)
        # self.positional_wise = None
        self.feed_forward = PostionWiseFeedForward(d_model, d_ff, dropout)
    
    def forward(self, x: torch.Tensor, mask):
        
        # shape(mha) = [batch_size x seq_len x d_model]
        # shape(encoder_attention_weights) = [batch_size x num_heads x seq_len x seq_len]
        mha, encoder_attention_weights = self.mha(x, x, x, mask=mask)
        
        # shape(norm1) = [batch_size x seq_len x d_model]
        norm_1 = self.norm_1(mha, x)
        
        # shape(feed_forward) = [batch_size x seq_len x d_model]
        feed_forward = self.feed_forward(norm_1)
        
        # shape(norm_2) = [batch_size x seq_len x d_model]
        norm_2 = self.norm_2(feed_forward, norm_1)
        
        return norm_2, encoder_attention_weights

# %% ../../nbs/transformer/07_transformer.encoder.ipynb 9
class Encoder(nn.Module):
    def __init__(
        self, embedding: Callable, d_model: int, num_heads: int, num_layers: int,
        d_ff: int, dropout: float = 0.3
    ):
        super().__init__()
        self.embedding = embedding
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoders = nn.ModuleList([
            EncoderLayer(
                d_model, num_heads, d_ff, dropout
            ) for layer in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        # shape(x) = [batch_size x src_seq_len]
        
        # shape(embeddings) = [batch_size x src_seq_len x d_model]
        embeddings = self.embedding(x)
        # shape(encoding) = [batch_size x src_seq_len x d_model]
        encoding = self.positional_encoding(embeddings)
        
        for encoder in self.encoders:
            # shape(encoding) = [batch_size x src_seq_len x d_model]
            # shape(encoder_attention_weights) = [batch_size x num_heads x src_seq_len x src_seq_len]
            encoding, encoder_attention_weights = encoder(encoding, mask)
        
        return encoding, encoder_attention_weights