import torch
from torch import nn

class MLPBlock(nn.Module):
    def __init__(self, 
                 embed_dim = 768,
                 hidden_size = 3978, 
                 dropout = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape = embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features = embed_dim, 
                      out_features = hidden_size),
            nn.GELU(),
            nn.Dropout(p = dropout),
            nn.Linear(in_features = hidden_size, 
                      out_features = embed_dim),
            nn.GELU(),
            nn.Dropout(p = dropout)
        )
    def forward(self, x):
        x = self.norm(x)
        x = self.mlp(x)
        return x
