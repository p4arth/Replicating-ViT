import torch
from torch import nn
from msa_block import MSABlock
from mlp_block import MLPBlock

class TransformerEncoder(nn.Module):
    def __init__(self,
                 embed_dim = 768, 
                 attn_heads = 12, 
                 attn_dropout = 0, 
                 mlp_hidden_size = 3978,
                 mlp_dropout = 0.1):
        super().__init__()
        self.msa = MSABlock(
            embed_dim = embed_dim,
            heads = attn_heads,
            attn_dropout = attn_dropout
        )
        self.mlp = MLPBlock(
            embed_dim = embed_dim, 
            hidden_size = mlp_hidden_size, 
            dropout = mlp_dropout
        )

    def forward(self, x):
        x = self.msa(x) + x
        x = self.mlp(x) + x
        return x
