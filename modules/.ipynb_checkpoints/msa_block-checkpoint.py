import torch
from torch import nn

class MSABlock(nn.Module):
    def __init__(self, 
                 embed_dim = 768, 
                 heads = 12, 
                 attn_dropout = 0):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape = embed_dim)
        self.multihead_attention = nn.MultiheadAttention(embed_dim = embed_dim, 
                                                         num_heads = heads, 
                                                         dropout = attn_dropout,
                                                         batch_first = True)
    def forward(self, x):
        x = self.norm(x)
        x, _ = self.multihead_attention(query = x,
                                        key = x,
                                        value = x,
                                        need_weights = False)
        return x
