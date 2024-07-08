import torch.nn as nn
from torch import Tensor
import torch
from src.nn.reference_transformers import Transformer


class DotProductLinkPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src: Tensor, dst: Tensor) -> Tensor:
        edge_scores = (src * dst).sum(dim=1)
        return torch.sigmoid(edge_scores)


class LinearLinkPredictor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = nn.Linear(in_channels, in_channels)
        self.lin_dst = nn.Linear(in_channels, in_channels)
        self.lin_final = nn.Linear(in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src: Tensor, dst: Tensor) -> Tensor:
        # Apply linear transformations and sum
        h = self.lin_src(src) + self.lin_dst(dst)
        h = torch.relu(h)

        # Final linear transformation and apply sigmoid
        return self.sigmoid(self.lin_final(h))


class BilinearLinkPredictor(nn.Module):
    def __init__(self, in_channels, edge_types):
        super().__init__()
        self.edge_types = edge_types
        self.bilinear_layer = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, src: Tensor, dst: Tensor) -> Tensor:
        # Apply bilinear transformation via linear layer without bias
        transformed_src = self.bilinear_layer(src)

        # Compute the interaction scores using element-wise multiplication and sum
        interaction_scores = (transformed_src * dst).sum(dim=1)

        # Apply sigmoid to get probabilities
        return torch.sigmoid(interaction_scores)


class TransformerLinkPredictor(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 num_heads: int,
                 dropout: float,
                 edge_types: list,
                 bias: bool,
                 mlp_expansion=2,
                 ):
        super().__init__()
        self.edge_types = edge_types

        self.transformer = Transformer(emb_dim=emb_dim,
                                       dropout=dropout,
                                       expansion_factor=mlp_expansion,
                                       num_heads=num_heads,
                                       bias=bias
                                       )
        self.lin = nn.Linear(emb_dim, 1)

    def forward(self, src: Tensor, dst: Tensor) -> Tensor:
        h = self.transformer.forward(k=src, v=src, q=dst)
        return torch.sigmoid(self.lin(h))
