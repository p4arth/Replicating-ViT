import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, 
                 in_channels = 3,
                 out_channels = 768,
                 patch_size = 16):
        super().__init__()
        self.patch_size = patch_size
        self.patcher = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = patch_size,
            stride = patch_size,
            padding = 0
        )
        self.flat = nn.Flatten(start_dim = 2, end_dim = 3)

    def forward(self, x):
        assert x.shape[2] % self.patch_size == 0, 'Img. size not divisible by patch size'
        assert x.shape[3] % self.patch_size == 0, 'Img. size not divisible by patch size'
        assert x.ndim == 4, 'Array must be 4-Dimensional'
        x = self.patcher(x)
        x = self.flat(x)
        return x.permute(0,2,1)
