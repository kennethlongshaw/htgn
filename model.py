import torch.nn as nn
import torch_geometric as geo
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.conv import MessagePassing
from typing import Union, Tuple, Final
from torch import Tensor
from torch_geometric.typing import OptTensor
import torch.nn.functional as F
from torch_geometric.utils import scatter
from utils import filter_by_index
import pytorch_lightning as pl

import torch
from typing import Optional, Callable, Any
from dataclasses import dataclass, asdict


class FlashTransformerConv(geo.nn.TransformerConv):
    """
        Implements the scaled dot product attention function in torch that can utilize flash attention or whatever
        most efficient implementation is available
    """

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


class GraphTransformerModel(BasicGNN):
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int,
                  **kwargs) -> MessagePassing:

        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', False)
        edge_dim = kwargs.pop('edge_dim', None)
        beta = kwargs.pop('beta', True)
        bias = kwargs.pop('bias', True)
        root_weight = kwargs.pop('root_weight', True)

        # Do not use concatenation in case the layer `TransformerConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'TransformerConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        return geo.nn.TransformerConv(in_channels=in_channels,
                                      out_channels=out_channels,
                                      heads=heads,
                                      concat=concat,
                                      edge_dim=edge_dim,
                                      beta=beta,
                                      bias=bias,
                                      root_weight=root_weight,
                                      dropout=self.dropout.p, **kwargs)


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
        src = self.src_linear(src)
        dst = self.dst_linear(dst)

        # Use src as both key and value in the attention mechanism
        x, self.attn_weight = self.attn(query=dst.unsqueeze(0), key=src.unsqueeze(0), value=src.unsqueeze(0),
                                        is_causal=False, attn_mask=None
                                        )
        x = x.squeeze(0)  # Remove the batch dimension added for multihead attention
        x = self.layer_norm(x)
        x = x + self.mlp(x)

        return x


@dataclass
class GraphTransformerConfig:
    in_channels: int | tuple
    hidden_channels: int
    num_layers: int
    out_channels: Optional[int] = None
    dropout: Optional[float] = None
    act: Optional[str | Callable] = 'relu'
    act_first: Optional[bool] = None
    act_kwargs: Optional[dict] = None
    norm: Optional[str | Callable] = None
    norm_kwargs: Optional[dict[str, Any]] = None
    jk: Optional[str] = None


class SimpleMessageEncoder(pl.LightningModule):
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

        # entities are NODE, and EDGE
        self.entity_emb_layer = nn.Embedding(num_embeddings=2, embedding_dim=emb_dim)

        # actions are CREATE, UPDATE, DELETE
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
                #entity_type: list[int],
                #action_type: list[int],

                src_node_types: list[int],
                src_features: list[Tensor],

                edge_types: list[int],
                edge_features: list[Tensor],

                dst_node_types: list[int],
                dst_features: list[Tensor],
                ):
        """
        :param action_type:
        :param entity_type:
        :param src_node_types: list of ints indicating node type
        :param src_features: list of ragged tensors dependent on src node type

        :param edge_types: list of ints indicating edge type
        :param edge_features: tensor of ints indicating edge type

        :param dst_node_types: list of ints indicating node type
        :param dst_features: list of ragged tensors dependent on dst node type
        :return:
        """
        # Learning Embedding
        #entity_emb = self.entity_emb[entity_type]
        #action_emb = self.entity_emb[action_type]


        # concat all node data to normalize at once
        concat_node_features = [('src', i, src) for i, src in enumerate(src_features)] + [('dst', j, dst) for j, dst in
                                                                                          enumerate(dst_features)]
        concat_node_types = src_node_types + dst_node_types

        norm_src_features = {}
        norm_dst_features = {}

        # norm node features
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

        # norm edge features
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
            src_with_edge_f = torch.cat([src_f, edge_f], dim=1)

            messages[idx] = self.edge_encoders[k[1]](src=src_with_edge_f, dst=dst_f)

        return messages


class MessageEncoder(pl.LightningModule):
    def __init__(self,
                 gnn_kwargs,
                 metadata: tuple,
                 emb_dim: int,
                 node_dims: list,
                 edge_dims: list,
                 dropout: float,
                 n_head: 1 = int,
                 mlp_expansion_factor: 2 = int,
                 bias: False = bool,
                 ):
        super().__init__()

        self.dropout = dropout
        self.emb_dim = emb_dim

        # entities are NODE, and EDGE
        self.entity_emb_layer = nn.Embedding(num_embeddings=2, embedding_dim=emb_dim)

        # actions are CREATE, UPDATE, DELETE
        self.action_emb_layer = nn.Embedding(num_embeddings=3, embedding_dim=emb_dim)

        # one for each edge type
        self.edge_emb_layer = nn.Embedding(num_embeddings=len(edge_dims), embedding_dim=emb_dim)

        # one for each node type
        self.node_emb_layer = nn.Embedding(len(node_dims), emb_dim)

        self.node_norms = [nn.InstanceNorm1d(d, track_running_stats=True) for d in node_dims]
        self.node_emd_norm = nn.LayerNorm(emb_dim)

        self.edge_norms = [nn.InstanceNorm1d(d, track_running_stats=True) for d in edge_dims]
        self.edge_emb_norm = nn.LayerNorm(emb_dim)

        self.action_norm = nn.LayerNorm(emb_dim)
        self.entity_norm = nn.LayerNorm(emb_dim)

        # self.encoders = [MessageTransformer(emb_dim=emb_dim,
        #                                     bias=bias,
        #                                     dropout=dropout,
        #                                     n_head=n_head,
        #                                     expansion_factor=mlp_expansion_factor
        #                                     ) for _ in edge_dims]

        self.encoder = geo.nn.to_hetero(GraphTransformerModel(**asdict(gnn_kwargs)),
                                        metadata=metadata,
                                        aggr='sum')

    def badforward(self,
                   entity_type: Tensor,
                   action_type: Tensor,
                   dst_node_id: Tensor,
                   src_node_id: Tensor,
                   dst_node_types: Tensor,
                   dst_features: list[Tensor],
                   src_features: list[Tensor],
                   src_node_types: Tensor,
                   edge_features: list[Tensor],
                   edge_types: Tensor,
                   ):

        """
            Embedding Layer: look up all entity, action, node type, edge type by index
            Norm Layer: apply to all embedding and node/edge features

        """

        # shape check on all
        num_records = entity_type.shape[0]
        assert all(tensor.size(0) == num_records for tensor in [
            dst_node_id, dst_node_types, dst_features,
            src_features, src_node_types, edge_features, edge_types
        ]), "All input tensors must have the same number of records"

        # Learning Embedding
        entity_emb = self.entity_emb[entity_type]
        action_emb = self.entity_emb[action_type]
        src_node_emb = self.node_emb[src_node_types]
        dst_node_emb = self.node_emb[dst_node_types]
        edge_emb = self.edge_emb(edge_types)

        # concat all node data to normalize at once
        concat_node_features = [(i, src) for i, src in enumerate(src_features)] + [(j, dst) for j, dst in
                                                                                   enumerate(dst_features)]
        concat_node_types = src_node_types.tolist() + dst_node_types.tolist()

        node_feature_dict = {}

        # norm node features
        for type_id, _ in enumerate(self.node_norms):
            type_data = filter_by_index(concat_node_features, concat_node_types, type_id)
            if len(type_data) > 0:
                type_index = [f[0] for f in type_data]
                type_features = torch.cat([f[1] for f in type_data], dim=0)
                node_type_features = self.node_norms[type_id](type_features)
                node_feature_dict[type_id] = {'index': type_index, 'features': node_type_features}

        edge_feature_dict = {}
        edge_features = [(i, f) for i, f in enumerate(edge_features)]
        # norm edge features
        for type_id, _ in enumerate(self.edge_norms):
            type_data = filter_by_index(edge_features, edge_types.tolist(), type_id)
            if len(type_data) > 0:
                type_index = [f[0] for f in type_data]
                type_features = torch.cat([f[1] for f in type_data], dim=0)
                edge_type_features = self.edge_norms[type_id](type_features)
                edge_feature_dict[type_id] = {'index': type_index, 'features': edge_type_features}

        # Initialize messages
        messages = torch.zeros_like(dst_node_emb)

        for src_type, edge_type, dst_type in set(
                zip(src_node_types.tolist(), edge_types.tolist(), dst_node_types.tolist())):
            src_node_features = node_feature_dict[src_type]['features']
            dst_node_features = node_feature_dict[dst_type]['features']

        # Apply the appropriate transformer based on edge type
        for edge_id, _ in enumerate(self.encoders):
            mask = edge_types == edge_id
            if mask.any():
                messages[mask] = self.encoders[edge_id](src_aggregate[mask], dst_aggregate[mask])

        return dst_node_id, messages

    def oldforward(self,
                   entity_type: Tensor,
                   action_type: Tensor,
                   src_features: Tensor,
                   src_node_types: Tensor,
                   dst_node_id: Tensor,
                   dst_features: Tensor,
                   dst_node_types: Tensor,
                   edge_features: Tensor,
                   edge_types: Tensor,
                   time: Tensor):

        num_records = src_features.shape[0]

        concat_node_features = torch.cat((src_features, dst_features), dim=0)
        concat_node_types = torch.cat((src_node_types, dst_node_types), dim=0)

        for type_id, norm_fn in enumerate(self.node_norms):
            type_mask = (concat_node_types == type_id)
            masked_features = concat_node_features[type_mask]
            concat_node_features[type_mask] = norm_fn(masked_features)

        src_features, dst_features = concat_node_features[:num_records], concat_node_features[num_records:]

        for type_id, norm_fn in enumerate(self.edge_norms):
            type_mask = (edge_types == type_id)
            masked_features = edge_features[type_mask]
            edge_features[type_mask] = norm_fn(masked_features)

        # norm emb layers, so they are aligned before being passed through diff transformers
        src_node_emb = self.node_emb_layer[src_node_types]
        dst_node_emb = self.node_emb_layer[dst_node_types]

        src_node_emb = self.node_emd_norm(src_node_emb)
        dst_node_emb = self.node_emd_norm(dst_node_emb)

        edge_emb = self.edge_emb_layer[edge_types]

        entity_emb = self.entity_emb_layer[entity_type]
        entity_emb = self.entity_norm(entity_emb)

        action_emb = self.action_emb_layer[action_type]
        action_emb = self.action_norm(action_emb)

        src = torch.cat(tensors=[src_features, src_node_emb,
                                 edge_features, edge_emb,
                                 entity_emb, action_emb], dim=-1)
        dst = torch.cat(tensors=[dst_features, dst_node_emb], dim=-1)

        messages = torch.zeros(num_records, self.emb_dim)

        for type_id, attn_fn in enumerate(self.encoders):
            type_mask = (edge_types == type_id)
            messages[type_mask] = attn_fn(src[type_mask], dst[type_mask])

        return dst_node_id, messages, time


class SumAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce='sum')
