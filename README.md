## Replicating-ViT

### This Repository is an implementation of [An Image is worth 16x16 words](https://arxiv.org/abs/2010.11929). A paper that came out in 2020 which used the highly successful transformer models from natural language processing for computer vision tasks. This implementation is purely in PyTorch.

## Table of Contents

- Getting Started
- Usage
- Module details

## Getting Started

The ViT model would require an installation of [PyTorch](https://pytorch.org/) to run.

To clone this repository locally use the following command in the CLI:

```
!git clone https://github.com/p4arth/Replicating-ViT.git
```

## Usage

Import the ViT (Vision Transformer) module which is under ```modules.vit```.

```python
from modules.vit import ViT
# Initializing the model
model = ViT()
```

## Module details 

The [modules](https://github.com/p4arth/Replicating-ViT/tree/main/modules) folder contains 5 submodules that altogether form the vision transformer model.

- **[Patch Embeddings](https://github.com/p4arth/Replicating-ViT/blob/main/modules/patch_embedding.py)**

  This module contains the patch embeddings class which is used in the paper to turn and image into patches of size 16x16. The patch embeddings are then flattened and passed onto the transformer encoder block.

- **[Multi-Headed Self Attention](https://github.com/p4arth/Replicating-ViT/blob/main/modules/msa_block.py)** (MSA)

  This module contains the multi-headed self attention block which resides inside the transformer encoder. The block applies a series of attention heads to the input provided.

- **[Multi-Layer Perceptron](https://github.com/p4arth/Replicating-ViT/blob/main/modules/mlp_block.py)** (MLP)

  This module proceeds the multi-headed self attention block and contains a multi-layer perceptron, also called a dense layer.

- **[Transformer Encoder](https://github.com/p4arth/Replicating-ViT/blob/main/modules/transformer_encoder.py)**

  This module combines both MSA and the MLP block to together form the transformer encoder layer. The input to this layer is flattened patches of an image which goes through a series of transformation in the MSA and MLP blocks.

- **[ViT](https://github.com/p4arth/Replicating-ViT/blob/main/modules/vit.py)** 

  This module implements the Vision Transformer model.



 
