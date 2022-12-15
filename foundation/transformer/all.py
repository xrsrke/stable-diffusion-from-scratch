# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/transformer/07a_transformer.ipynb.

# %% auto 0
__all__ = ['Transformer']

# %% ../../nbs/transformer/07a_transformer.ipynb 4
import torch
from torch import nn

from .encoder import EncoderLayer, Encoder
from .embedding import PositionalEncoding, TextEmbedding

# %% ../../nbs/transformer/07a_transformer.ipynb 32
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_len: int, trg_vocab_len: int,
        d_model: int, d_ff: int,
        num_layers: int, num_heads: int,
        src_pad_idx: float, trg_pad_idx: float,
        dropout: float = 0.3,
        efficient_mha: bool = False
    ):
        super().__init__()
        self.num_heads = num_heads
        self.efficient_mha = efficient_mha
        
        encoder_embedding = TextEmbedding(
            vocab_size=src_vocab_len,
            padding_idx=src_pad_idx,
            d_model=d_model
        )
        decoder_embedding = TextEmbedding(
            vocab_size=trg_vocab_len,
            padding_idx=trg_pad_idx,
            d_model=d_model
        )
        
