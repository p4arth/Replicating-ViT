import torch
from torch import nn
from modules.transformer_encoder import TransformerEncoder
from modules.patch_embedding import Patch_Embedding

class ViT(nn.Module):
    def __init__(self,
                 image_size = 224,
                 patch_size = 16,
                 embed_dim = 768,
                 in_channels = 3,
                 transformer_layers = 12,
                 attn_heads = 12, 
                 attn_dropout = 0, 
                 mlp_hidden_size = 3978,
                 mlp_dropout = 0.1, 
                 num_classes = 3, 
                 embedding_dropout = 0.1):
        super().__init__()
        self.num_patches = (image_size * image_size) // patch_size**2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim),
                                      requires_grad = True)
        self.positional_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim),
            requires_grad = True
        )
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.patcher = PatchEmbedding(in_channels = in_channels,
                                      out_channels = embed_dim,
                                      patch_size = patch_size)
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoder(embed_dim = embed_dim, 
                                 attn_heads = attn_heads, 
                                 attn_dropout = attn_dropout, 
                                 mlp_hidden_size = mlp_hidden_size, 
                                 mlp_dropout = mlp_dropout) for _ in range(transformer_layers)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape = embed_dim),
            nn.Linear(in_features = embed_dim, 
                      out_features = num_classes)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.cls_token.expand(batch_size, -1, -1)
        x = self.patcher(x)
        x = torch.cat((class_token, x), dim = 1)
        x = x + self.positional_embeddings
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])
        return x
