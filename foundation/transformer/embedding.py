# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/07e_transformer.embedding.ipynb.

# %% auto 0
__all__ = ['TextEmbedding', 'PositionalEncoding']

# %% ../../nbs/07e_transformer.embedding.ipynb 4
import math

import torch
from torch import nn

# %% ../../nbs/07e_transformer.embedding.ipynb 7
class TextEmbedding(nn.Module):
    def __init__(self, vocab_size: int, padding_idx, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
    
    def forward(self, x):
        # shape(x) = [batch_size x seq_len]
        # shape(embedding) = [batch_size x seq_len x d_model]
        embedding = self.embed(x)
        return embedding * math.sqrt(self.d_model)

# %% ../../nbs/07e_transformer.embedding.ipynb 14
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.3, max_seq_len : float = 2000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        two_i = torch.arange(0, d_model, step=2).float()
        div_term = torch.pow(1000, (two_i/torch.tensor([d_model]))).float()
        
        pe[:, 0::2] = torch.sin(pos/div_term)
        pe[:, 1::2] = torch.cos(pos/div_term)
        
        # add one dim for batch_size
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor):
        # x is text embedding
        # shape(x) = [batch_size x seq_len x d_model]
        seq_len = x.shape[1]
        
        # extract the position for seq_len
        pe = self.pe[:, :seq_len].detach()
        
        x = x.add(pe)
        
        return self.dropout(x)
