import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import scatter
from utils import filter_by_index
import pytorch_lightning as pl
import torch


class MessageTransformer(nn.Module):
    """
        Implements a transformer block for message cross attention in order to aggregate messages to their destination
    """

    def __init__(self,
                 emb_dim: int,
                 bias: bool = True,
                 dropout: float = 0.2,
                 n_head: int = 1,
                 expansion_factor: int = 2):
        super().__init__()
        self.src_linear = nn.LazyLinear(emb_dim)
        self.dst_linear = nn.LazyLinear(emb_dim)
        self.attn = nn.MultiheadAttention(num_heads=n_head,
                                          embed_dim=emb_dim,
                                          bias=bias,
                                          dropout=dropout
                                          )
        self.layer_norm = nn.LayerNorm(emb_dim, bias=bias)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * expansion_factor, bias=bias),
            nn.ReLU(),
            nn.Linear(emb_dim * expansion_factor, emb_dim, bias=bias),
            nn.Dropout(dropout)
        )

        self.attn_weight = None

    def forward(self, src, dst):
        # align dimensions
        src = self.src_linear(src)
        dst = self.dst_linear(dst)

        # Use src as both key and value in the attention mechanism
        x, self.attn_weight = self.attn(query=dst.unsqueeze(0), key=src.unsqueeze(0), value=src.unsqueeze(0),
                                        is_causal=False, attn_mask=None
                                        )
        x = x.squeeze(0)  # Remove the batch dimension added for multi-head attention
        x = self.layer_norm(x)
        x = x + self.mlp(x)

        return x


class MessageEncoder(pl.LightningModule):
    def __init__(self,
                 emb_dim: int,
                 node_dims: list,
                 edge_dims: list,
                 dropout: float,
                 n_head: int = 1,
                 mlp_expansion_factor: int = 2,
                 bias: bool = False,
                 ):
        super().__init__()

        self.dropout = dropout
        self.emb_dim = emb_dim

        # entities are 0 NODE, and 1 EDGE
        self.entity_emb_layer = nn.Embedding(num_embeddings=2, embedding_dim=emb_dim)

        # actions are 0 CREATE, 1 UPDATE, 2 DELETE
        self.action_emb_layer = nn.Embedding(num_embeddings=3, embedding_dim=emb_dim)

        # norms
        self.action_norm = nn.LayerNorm(emb_dim)
        self.entity_norm = nn.LayerNorm(emb_dim)

        self.node_norms = torch.nn.ModuleList([nn.BatchNorm1d(d) for d in node_dims])

        self.edge_norms = torch.nn.ModuleList([nn.BatchNorm1d(d) for d in edge_dims])

        self.edge_encoders = torch.nn.ModuleList([MessageTransformer(emb_dim=emb_dim,
                                                                     bias=bias,
                                                                     dropout=dropout,
                                                                     n_head=n_head,
                                                                     expansion_factor=mlp_expansion_factor)
                                                  for _ in edge_dims])

    def forward(self,
                entity_types: Tensor,
                action_types: Tensor,

                src_node_types: list[int],
                src_features: list[Tensor],

                edge_types: list[int],
                edge_features: list[Tensor],

                dst_node_types: list[int],
                dst_features: list[Tensor],
                ):
        """
        :param action_types: Tensor of int ids for action codes
        :param entity_types: Tensor of int ids for entity codes
        :param src_node_types: list of ints indicating node type
        :param src_features: list of ragged tensors dependent on src node type

        :param edge_types: list of ints indicating edge type
        :param edge_features: tensor of ints indicating edge type

        :param dst_node_types: list of ints indicating node type
        :param dst_features: list of ragged tensors dependent on dst node type
        :return:
        """
        
        # Learning Embedding
        entity_emb = self.entity_emb_layer(entity_types)
        action_emb = self.action_emb_layer(action_types)

        # concat all node data to normalize at once
        concat_node_features = [('src', i, src) for i, src in enumerate(src_features)] + [('dst', j, dst) for j, dst in
                                                                                          enumerate(dst_features)]
        concat_node_types = src_node_types + dst_node_types

        norm_src_features = {}
        norm_dst_features = {}

        # norm node features by types
        for type_id, _ in enumerate(self.node_norms):
            type_data = filter_by_index(concat_node_features, concat_node_types, type_id)
            if len(type_data) > 0:
                direction = [f[0] for f in type_data]
                og_index = [f[1] for f in type_data]
                type_features = torch.stack([f[2] for f in type_data])
                norm_type_features = self.node_norms[type_id](type_features)
                for node_dir, idx, feat in zip(direction, og_index, norm_type_features):
                    match node_dir:
                        case 'src':
                            norm_src_features[idx] = feat
                        case 'dst':
                            norm_dst_features[idx] = feat

        # norm edge features by types
        norm_edge_feature_dict = {}
        edge_features = [(i, f) for i, f in enumerate(edge_features)]
        for type_id, _ in enumerate(self.edge_norms):
            type_data = filter_by_index(edge_features, edge_types, type_id)
            if len(type_data) > 0:
                og_index = [f[0] for f in type_data]
                type_features = torch.stack([f[1] for f in type_data])
                norm_type_features = self.edge_norms[type_id](type_features)
                for idx, feat in zip(og_index, norm_type_features):
                    norm_edge_feature_dict[idx] = feat

        # align features before encoding
        message_dict = {}
        for i, edge_type in enumerate(zip(src_node_types, edge_types, dst_node_types)):
            features = norm_src_features[i], norm_edge_feature_dict[i], norm_dst_features[i]
            if edge_type not in message_dict:
                message_dict[edge_type] = {i: features}
            else:
                message_dict[edge_type][i] = features

        # final message encoding
        messages = torch.zeros(len(src_features), self.emb_dim)
        for k, feature_dict in message_dict.items():
            idx = torch.tensor(list(feature_dict.keys()))
            src_f = torch.stack([i[0] for i in feature_dict.values()])
            edge_f = torch.stack([i[1] for i in feature_dict.values()])
            dst_f = torch.stack([i[2] for i in feature_dict.values()])
            concat_src = torch.cat([src_f, edge_f, entity_emb[idx], action_emb[idx]], dim=1)

            messages[idx] = self.edge_encoders[k[1]](src=concat_src, dst=dst_f)

        return messages


class SumAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce='sum')
