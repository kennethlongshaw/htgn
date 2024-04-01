from typing import Optional, Any
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch_geometric as geo
from torch import Tensor
from torch_geometric.typing import OptTensor
import torch.nn.functional as F


class FlashTransformerConv(geo.nn.TransformerConv):
    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key_j = key_j + edge_attr
            value_j = value_j + edge_attr

        attn_output, alpha = F.scaled_dot_product_attention(
            query=query_i, key=key_j, value=value_j,
            attn_mask=None, dropout_p=self.dropout, is_causal=False
        )

        self._alpha = alpha.view(-1, self.heads)

        return attn_output


class MessageTransformer(nn.Module):
    def __init__(self,
                 src_dim: int,
                 edge_dim: int,
                 dst_dim: int,
                 emb_dim: int,
                 bias: bool,
                 dropout: float,
                 n_head: int,
                 expansion_factor: int):
        super().__init__()
        # Batch norm for raw input
        self.src_norm = nn.InstanceNorm1d(src_dim)
        self.dst_norm = nn.InstanceNorm1d(dst_dim)
        self.edge_norm = nn.InstanceNorm1d(edge_dim)

        self.src_emb_norm = nn.LayerNorm(emb_dim)
        self.dst_emb_norm = nn.LayerNorm(emb_dim)
        self.edge_emb_norm = nn.LayerNorm(emb_dim)

        self.action_norm = nn.LayerNorm(emb_dim)
        self.entity_norm = nn.LayerNorm(emb_dim)

        self.attn = nn.MultiheadAttention(num_heads=n_head,
                                          embed_dim=emb_dim,
                                          bias=bias,
                                          dropout=dropout
                                          )
        # Layer norm after representation learning
        self.layer_norm = nn.LayerNorm(emb_dim, bias=bias)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * expansion_factor, bias=bias),
            nn.ReLU(),
            nn.Linear(emb_dim * expansion_factor, emb_dim, bias=bias),
            nn.Dropout(dropout)
        )

        self.attn_weight = None

    def forward(self, src, src_emb, dst, dst_emb, edge, edge_emb, action_emb, entity_emb):
        # batch norm raw tensors
        src, dst, edge = self.src_norm(src), self.dst_norm(dst), self.edge_norm(edge)

        # layer norm emb tensors
        src_emb, dst_emb, edge_emb = self.src_emb_norm(src_emb), self.dst_emb_norm(dst_emb), self.edge_emb_norm(
            edge_emb)
        action_emb, entity_emb = self.action_norm(action_emb), self.entity_norm(entity_emb)

        # cat all into src and dst
        src = torch.cat(tensors=[src, src_emb,
                                 edge, edge_emb,
                                 entity_emb, action_emb], dim=-1)
        dst = torch.cat(tensors=[dst, dst_emb], dim=-1)

        # Use src as both key and value in the attention mechanism
        x, self.attn_weight = self.attn(query=dst.unsqueeze(0), key=src.unsqueeze(0), value=src.unsqueeze(0),
                                        is_causal=False, attn_mask=None
                                        )
        x = x.squeeze(0)  # Remove the batch dimension added for multihead attention
        x = self.layer_norm(x)
        x = x + self.mlp(x)

        return x


class MessageEncoder(pl.LightningModule):
    def __init__(self,
                 edges: list,
                 emb_dim: int,
                 node_dims: dict,
                 edge_dims: dict,
                 dropout: float,
                 n_head: 1 = int,
                 mlp_expansion_factor: 2 = int,
                 bias: False = bool
                 ):
        super().__init__()
        # entities are NODE, and EDGE
        self.entity_emb_layer = nn.Embedding(num_embeddings=2, embedding_dim=emb_dim)

        # actions are CREATE, UPDATE, DELETE
        self.action_emb_layer = nn.Embedding(num_embeddings=3, embedding_dim=emb_dim)

        # one for each edge type
        self.edge_emb_layer = nn.Embedding(num_embeddings=len(edges), embedding_dim=emb_dim)

        # one for each node type
        self.nodes = list(set([e[0] for e in edges] + [f[2] for f in edges]))
        self.node_emb_layer = nn.Embedding(len(self.nodes), emb_dim)

        self.encoders = {e: MessageTransformer(src_dim=node_dims[e[0]],
                                               dst_dim=node_dims[e[2]],
                                               edge_dim=edge_dims[e],
                                               emb_dim=emb_dim,
                                               bias=bias,
                                               dropout=dropout,
                                               n_head=n_head,
                                               expansion_factor=mlp_expansion_factor
                                               ) for e in edges}

    def forward(self,
                entity_type,
                action_type,
                src_id,
                src_features,
                dst_id,
                dst_features,
                edge_id,
                edge_features,
                edge_type,
                time):
        entity_emb = self.entity_emb_layer(entity_type)
        action_emb = self.action_emb_layer(action_type)
        src_node_emb = self.node_emb_layer(torch.cat(edge_type[0]))
        dst_node_emb = self.node_emb_layer(torch.cat(edge_type[2]))
        edge_emb = self.edge_emb_layer(edge_type)

        # match records by edge type and pass them through associated message transformer
        # like this
        # message = enc(src, src_emb, dst, dst_emb, edge, edge_emb, action_emb, entity_emb)
        # but then reassemble the batch in the same order

        return dst_id, message, time
