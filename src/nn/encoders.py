import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import scatter
import pytorch_lightning as pl
import torch
from dataclasses import dataclass
from src.utils.utils import filter_by_index
from src.nn.protocols import MemoryBatch

class MessageTransformer(nn.Module):
    """
        Implements a transformer block for message cross attention in context of the destination
    """

    def __init__(self,
                 emb_dim: int,
                 src_dim: int,
                 dst_dim: int,
                 bias: bool = True,
                 dropout: float = 0.2,
                 n_head: int = 1,
                 expansion_factor: int = 2):
        super().__init__()
        self.src_norm = nn.LayerNorm(src_dim, bias=bias)
        self.dst_norm = nn.LayerNorm(dst_dim, bias=bias)
        self.src_linear_k = nn.Linear(in_features=src_dim, out_features=emb_dim, bias=bias)
        self.src_linear_v = nn.Linear(in_features=src_dim, out_features=emb_dim, bias=bias)
        self.dst_linear = nn.Linear(in_features=dst_dim, out_features=emb_dim, bias=bias)
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

    def forward(self, src, dst, need_weights=False):
        # norm and align
        src_k = self.src_linear_k(self.src_norm(src))
        src_v = self.src_linear_v(self.src_norm(src))
        dst = self.dst_linear(self.dst_norm(dst))

        # Use src as both key and value in the attention mechanism
        attn_out, self.attn_weight = self.attn(query=dst.unsqueeze(0), key=src_k.unsqueeze(0), value=src_v.unsqueeze(0),
                                               is_causal=False, attn_mask=None, need_weights=need_weights
                                               )
        x = dst + attn_out.squeeze(0)  # Remove the batch dimension added for multi-head attention
        x = x + self.mlp(self.layer_norm(x))

        return x


@dataclass
class HeteroMessageEncoder_Config:
    emb_dim: int
    node_dims: list[int]
    edge_dims: list[int]
    memory_dim: int
    time_dim: int
    edge_types: list[tuple[int, int, int]]
    dropout: float
    n_head: int = 1
    mlp_expansion_factor: int = 2
    bias: bool = False
    entity_dim: int = 2
    action_dim: int = 3


class HeteroMessageEncoder(pl.LightningModule):
    """
        Implements a Message Transformer for each edge type of a graph.
    """

    def __init__(self,
                 cfg: HeteroMessageEncoder_Config
                 ):
        super().__init__()

        self.dropout = cfg.dropout
        self.emb_dim = cfg.emb_dim

        # entities are 0 NODE, and 1 EDGE
        self.entity_emb_layer = nn.Embedding(num_embeddings=cfg.entity_dim, embedding_dim=cfg.emb_dim)

        # actions are 0 CREATE, 1 UPDATE, 2 DELETE
        self.action_emb_layer = nn.Embedding(num_embeddings=cfg.action_dim, embedding_dim=cfg.emb_dim)

        # norms
        self.action_norm = nn.LayerNorm(cfg.emb_dim)
        self.entity_norm = nn.LayerNorm(cfg.emb_dim)

        # emb layer for each node and edge type to capture schema structure
        self.node_emb_layers = nn.Embedding(num_embeddings=len(cfg.node_dims), embedding_dim=cfg.emb_dim)
        self.edge_emb_layers = nn.Embedding(num_embeddings=len(cfg.edge_dims), embedding_dim=cfg.emb_dim)

        self.node_norms = nn.ModuleList([nn.LayerNorm(d + cfg.emb_dim) for d in cfg.node_dims])
        self.edge_norms = nn.ModuleList([nn.LayerNorm(d + cfg.emb_dim) for d in cfg.edge_dims])

        encoder_dims = [(cfg.node_dims[e[0]] +  # src node dim
                         cfg.memory_dim +
                         cfg.emb_dim +  # node type emb
                         cfg.edge_dims[e[1]] +
                         cfg.emb_dim +  # edge type emb
                         cfg.time_dim +
                         cfg.emb_dim +  # entity type emb
                         cfg.emb_dim,  # action type emb
                         cfg.node_dims[e[2]] + cfg.memory_dim + cfg.emb_dim)
                        for e in cfg.edge_types]

        # one encoder per edge
        self.edge_encoders = nn.ModuleList([MessageTransformer(emb_dim=cfg.emb_dim,
                                                               src_dim=src_dim,
                                                               dst_dim=dst_dim,
                                                               bias=cfg.bias,
                                                               dropout=cfg.dropout,
                                                               n_head=cfg.n_head,
                                                               expansion_factor=cfg.mlp_expansion_factor)
                                            for src_dim, dst_dim in encoder_dims])

    def forward(self,
                batch: MemoryBatch
                ):
        """
        :param rel_t_enc: Tensor of relative time encoding
        :param action_types: Tensor of int ids for action codes
        :param entity_types: Tensor of int ids for entity codes

        :param src_node_types: list of ints indicating node type
        :param src_features: list of ragged tensors dependent on src node type
        :param src_memories: Tensor of previous memory state for src node

        :param edge_types: list of ints indicating edge type
        :param edge_features: ragged tensors dependent on edge type

        :param dst_node_types: list of ints indicating node type
        :param dst_features: list of ragged tensors dependent on dst node type
        :param dst_memories: Tensor of previous memory state for dst node
        :return:
        """

        # Learning Embedding
        entity_emb = self.entity_emb_layer(batch.entity_types)
        action_emb = self.action_emb_layer(batch.action_types)

        # concat all node data to normalize at once
        concat_node_features = [('src', i, src) for i, src in enumerate(batch.src_features)] + \
                               [('dst', j, dst) for j, dst in enumerate(batch.dst_features)]
        concat_node_types = batch.src_node_types.tolist() + batch.dst_node_types.tolist()

        norm_src_features = {}
        norm_dst_features = {}

        # norm node features by types and organize into feature dicts
        # this is a bit messy because we are operating on list of ragged tensors
        for type_id in set(concat_node_types):
            type_data = filter_by_index(concat_node_features, concat_node_types, type_id)
            direction = [f[0] for f in type_data]
            og_index = [f[1] for f in type_data]
            type_features = torch.stack([f[2] for f in type_data])
            node_type_emb = self.node_emb_layers(torch.tensor(type_id)).repeat(type_features.shape[0], 1)

            if type_features.numel() > 0:
                type_features = torch.cat([type_features, node_type_emb], dim=1)
            else:
                type_features = node_type_emb

            node_norm_features = self.node_norms[type_id](type_features)
            for node_dir, idx, feat in zip(direction, og_index, node_norm_features):
                match node_dir:
                    case 'src':
                        norm_src_features[idx] = feat
                    case 'dst':
                        norm_dst_features[idx] = feat
                    case _:
                        raise ValueError(f'Invalid direction of {node_dir}')

        # norm edge features by types and organize into feature dicts
        # this is a bit messy because we are operating on list of ragged tensors
        norm_edge_feature_dict = {}
        edge_features = [(i, f) for i, f in enumerate(batch.edge_features)]
        for type_id in range(len(self.edge_norms)):
            type_data = filter_by_index(edge_features, batch.edge_types, type_id)
            if len(type_data) > 0:
                og_index = [f[0] for f in type_data]
                type_features = torch.stack([f[1] for f in type_data])
                edge_type_emb = self.edge_emb_layers(torch.tensor(type_id)).repeat(type_features.shape[0], 1)
                if type_features.numel() > 0:
                    type_features = torch.cat([type_features, edge_type_emb], dim=1)
                else:
                    type_features = edge_type_emb
                edge_norm_features = self.edge_norms[type_id](type_features)
                for idx, feat in zip(og_index, edge_norm_features):
                    norm_edge_feature_dict[idx] = feat

        # align features before encoding
        message_dict = {}
        for i, edge_name in enumerate(zip(batch.src_node_types,
                                          batch.edge_types,
                                          batch.dst_node_types)
                                      ):
            # edge_name format is (src node type, edge type, dst node type)
            features = norm_src_features[i], norm_edge_feature_dict[i], norm_dst_features[i]
            if edge_name not in message_dict:
                message_dict[edge_name] = {i: features}
            else:
                message_dict[edge_name][i] = features

        # final message encoding
        messages = torch.zeros(len(batch.src_features), self.emb_dim)
        for edge_name, feature_dict in message_dict.items():
            idx = torch.tensor(list(feature_dict.keys()))
            src_f = torch.stack([i[0] for i in feature_dict.values()])
            edge_f = torch.stack([i[1] for i in feature_dict.values()])
            dst_f = torch.stack([i[2] for i in feature_dict.values()])
            concat_src = torch.cat(tensors=
                                   [src_f,
                                    batch.src_memories[idx],
                                    edge_f,
                                    batch.rel_time_enc[idx],
                                    entity_emb[idx],
                                    action_emb[idx]
                                    ],
                                   dim=1)

            concat_dst = torch.cat([dst_f, batch.dst_memories[idx]], dim=1)

            messages[idx] = self.edge_encoders[edge_name[1]](src=concat_src, dst=concat_dst)

        return messages

    def draft_forward(self,
                      rel_t_enc: Tensor,
                      entity_types: Tensor,
                      action_types: Tensor,
                      src_node_types: Tensor,
                      src_features: list[Tensor],
                      src_memories: Tensor,
                      edge_types: Tensor,
                      edge_features: list[Tensor],
                      dst_node_types: Tensor,
                      dst_features: list[Tensor],
                      dst_memories: Tensor
                      ):
        # Learning Embedding
        entity_emb = self.entity_emb_layer(entity_types)
        action_emb = self.action_emb_layer(action_types)

        # Concat node features, types, and embeddings
        node_types = torch.cat([src_node_types, dst_node_types])
        node_type_emb = self.node_emb_layers(node_types)
        node_features_norm = []
        for norm, feat, emb in zip(self.node_norms, src_features + dst_features, node_type_emb):
            if feat.dim() > 0:
                print(feat.dim())
                print(feat)
                print(emb.expand(len(feat), -1))
                node_features_norm.append(norm(torch.cat([feat, emb.expand(feat.shape[0], -1)], dim=1)))
            else:
                node_features_norm.append(norm(emb))

        # Concat edge features, types, and embeddings
        edge_type_emb = self.edge_emb_layers([e[1] for e in edge_types])
        edge_features_norm = [norm(torch.cat([feat, emb.expand(feat.shape[0], -1)], dim=1) if feat.dim() > 0 else emb)
                              for norm, feat, emb in zip(self.edge_norms, edge_features, edge_type_emb)]

        # Prepare message inputs
        src_input = torch.cat([torch.stack(node_features_norm[:len(src_features)]),
                               src_memories,
                               torch.stack(edge_features_norm),
                               rel_t_enc,
                               entity_emb,
                               action_emb], dim=1)
        dst_input = torch.cat([torch.stack(node_features_norm[len(src_features):]), dst_memories], dim=1)

        # Encode messages
        messages = torch.zeros(len(src_features), self.emb_dim, device=entity_emb.device)
        for edge_type in range(len(self.edge_encoders)):
            idx = torch.where(edge_types == edge_type)[0]
            messages[idx] = self.edge_encoders[edge_type](src_input[idx], dst_input[idx])

        return messages


class ScatterAggregator(torch.nn.Module):
    def __init__(self,
                 reduce: str = 'sum',
                 impl: str = 'native'
                 ):
        super().__init__()
        valid_impls = ['native', 'geometric']
        assert impl in valid_impls, (
                'Specified implementation valid. Options are: ' + ', '.join(valid_impls)
        )
        self.impl = impl

        assert_msg = 'Specified reduce {reduce} for {impl} is invalid. Options are: '
        match impl:
            case 'native':
                valid_reduce_native = ("sum", "prod", "mean", "amax", "amin")
                assert reduce in valid_reduce_native, (
                        assert_msg.format(reduce=reduce, impl=impl) + ', '.join(valid_reduce_native))

            case 'geometric':
                valid_reduce_geo = ('sum', 'add', 'mul', 'mean', 'min', 'max')
                assert reduce in valid_reduce_geo, (
                        assert_msg.format(reduce=reduce, impl=impl) + ', '.join(valid_reduce_geo))

        self.reduce = reduce

    def forward(self, values: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        match self.impl:
            case 'geometric':
                return scatter(values, index, dim=0, dim_size=num_nodes, reduce=self.reduce)
            case 'native':
                # Initialize the output tensor to zeros with the appropriate shape
                reduced = torch.zeros(num_nodes, dtype=values.dtype, device=values.device)
                # Perform scatter reduce operation with 'sum' reduction
                return reduced.scatter_reduce_(dim=0, index=index, src=values, reduce=self.reduce)
            case _:
                assert NotImplementedError, f'Invalid implementation of {self.impl} specified'


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.lin = nn.Linear(1, out_channels)

    def forward(self, t: Tensor) -> Tensor:
        return self.lin(t.view(-1, 1)).cos()
