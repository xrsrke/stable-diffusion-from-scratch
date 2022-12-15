# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/transformer/07_transformer.decoder.ipynb.

# %% auto 0
__all__ = ['create_mask', 'DecoderLayer', 'Decoder']

# %% ../../nbs/transformer/07_transformer.decoder.ipynb 4
import torch
from torch import nn
from .attention import MultiHeadAttention
from .encoder import PostionWiseFeedForward
from .embedding import PositionalEncoding

# %% ../../nbs/transformer/07_transformer.decoder.ipynb 6
def create_mask(size):
    mask = torch.ones((1, size, size)).triu(1)

# %% ../../nbs/transformer/07_transformer.decoder.ipynb 11
class DecoderLayer(nn.ModuleList):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.3):
        super().__init__()
        self.norm_1 = ResidualLayerNorm(d_model)
        self.norm_2 = ResidualLayerNorm(d_model)
        self.norm_3 = ResidualLayerNorm(d_model)
        
        self.masked_mha = MultiHeadAttention(d_model, num_heads)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PostionWiseFeedForward(d_model, d_ff)
    
    def forward(self, x, encoder_outputs, trg_mask, src_mask):
        # shape(x) = [batch_size x trg_seq_len x d_model]
        # shape(encoder_output) = [batch_size x src_seq_len x d_model]
        
        
        # shape(masked_mha) = [batch_size x trg_se_len x d_model]
        # shape(masked_mha_attn_weights)
        # = [batch_size x num_heads x trg_seq_len x trg_seq_len]
        masked_mha, masked_mha_attn_weights = self.masked_mha(x, x, x, mask=trg_mask)
        
        norm_1 = self.norm_1(masked_mha, x)
        
        # shape(mha) = [batch_size x trg_seq_len x d_model]
        # shape(mha_attn_weights) = [batch_size x num_heads x trg_seq_len x trg_seq_len]
        mha, mha_attn_weights = self.mha(
            norm1, encoder_outputs, encoder_outputs,
            mask=src_mask
        )
        
        norm_2 = self.norm_2(mha, norm_1)
        
        feed_forward = self.feed_forward(norm_2)
        norm_3 = self.norm_3(feed_forward, norm_2)
        
        return norm_3, masked_mha_attn_weights, mha_attn_weights

# %% ../../nbs/transformer/07_transformer.decoder.ipynb 14
class Decoder(nn.Module):
    def __init__(
        self, embedding, d_model, num_heads, num_layers, d_ff,
        device="cpu", dropout=0.3
    ):
        super().__init__()
        self.embedding = embedding
        self.positional_encoding = PositionalEncoding(
            d_model, device=device
        )
        self.dropout = nn.Dropout(dropout)
        self.decoders = nn.ModuleList([
            DecoderLayer(
                d_model,
                num_heads,
                d_ff,
                dropout
            ) for layer in range(num_layers)
        ])
    
    def forward(self, x, encoder_output, trg_mask, src_mask):
        # shape(x) = [batch_size x trg_seq_len]
        
        # shape(embeddings) = [batch_size x trg_seq_len x d_model]
        embeddings = self.embedding(x)
        # shape(encoding) = [batch_size x trg_seq_len x d_model]
        encoding = self.positional_encoding(embedding)
        
        for decoder in self.decoders:
            # shape(encoding) = [batch_size x trg_seq_len x d_model]
            # shape(masked_mha_attn_weights) = [batch_size x num_heads x trg_seq_len x trg_seq_len]
            # shape(mha_attn_weights) = [batch_size x num_heads x trg_seq_len x src_seq_len]
            encoding, masked_mha_attn_weights, mha_attn_weights = decoder(
                encoding, encoder_output,
                trg_mask, src_mask
            )
        
        return encoding, masked_mha_attn_weights, mha_attn_weights
