import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import scatter
from utils import filter_by_index
import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class MessageTransformer(nn.Module):
    """
        Implements a transformer block for message cross attention in order to aggregate messages to their destination
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
        self.src_linear = nn.Linear(in_features=src_dim, out_features=emb_dim, bias=bias)
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

    def forward(self, src, dst):
        # norm and align
        src = self.src_linear(self.src_norm(src))
        dst = self.dst_linear(self.dst_norm(dst))

        # Determine whether to compute attention weights based on the mode
        need_weights = not self.training  # True if in eval mode, False if in training mode

        # Use src as both key and value in the attention mechanism
        attn_out, self.attn_weight = self.attn(query=dst.unsqueeze(0), key=src.unsqueeze(0), value=src.unsqueeze(0),
                                               is_causal=False, attn_mask=None, need_weights=need_weights
                                               )
        x = dst + attn_out.squeeze(0)  # Remove the batch dimension added for multi-head attention
        x = x + self.mlp(self.layer_norm(x))

        return x


class HeteroMessageEncoder(pl.LightningModule):
    def __init__(self,
                 emb_dim: int,
                 node_dims: list[int],
                 edge_dims: list[int],
                 edge_types: list[tuple[int, int, int]],
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

        # emb layer for each node and edge type to capture schema structure
        self.node_emb_layers = nn.Embedding(num_embeddings=len(node_dims), embedding_dim=emb_dim)
        self.edge_emb_layers = nn.Embedding(num_embeddings=len(edge_dims), embedding_dim=emb_dim)

        self.node_norms = nn.ModuleList([nn.LayerNorm(d + emb_dim) for d in node_dims])
        self.edge_norms = nn.ModuleList([nn.LayerNorm(d + emb_dim) for d in edge_dims])

        # src_f including emb dim, edge_f including emb dim, entity_emb[idx], action_emb[idx]]
        encoder_dims = [(node_dims[e[0]] + edge_dims[e[1]] + emb_dim * 4,
                         node_dims[e[2]] + emb_dim)
                        for e in edge_types]

        # one encoder per edge
        self.edge_encoders = nn.ModuleList([MessageTransformer(emb_dim=emb_dim,
                                                               src_dim=src_dim,
                                                               dst_dim=dst_dim,
                                                               bias=bias,
                                                               dropout=dropout,
                                                               n_head=n_head,
                                                               expansion_factor=mlp_expansion_factor)
                                            for src_dim, dst_dim in encoder_dims])

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
        :param edge_features: ragged tensors dependent on edge type

        :param dst_node_types: list of ints indicating node type
        :param dst_features: list of ragged tensors dependent on dst node type
        :return:
        """

        # Learning Embedding
        entity_emb = self.entity_emb_layer(entity_types)
        action_emb = self.action_emb_layer(action_types)

        # concat all node data to normalize at once
        concat_node_features = [('src', i, src) for i, src in enumerate(src_features)] + \
                               [('dst', j, dst) for j, dst in enumerate(dst_features)]
        concat_node_types = src_node_types + dst_node_types

        norm_src_features = {}
        norm_dst_features = {}

        # norm node features by types and organize into feature dicts
        # this is a bit messy because we are operating on list of ragged tensors
        for type_id, _ in enumerate(self.node_norms):
            type_data = filter_by_index(concat_node_features, concat_node_types, type_id)
            if len(type_data) > 0:
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
        edge_features = [(i, f) for i, f in enumerate(edge_features)]
        for type_id, _ in enumerate(self.edge_norms):
            type_data = filter_by_index(edge_features, edge_types, type_id)
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
        for i, edge_name in enumerate(zip(src_node_types, edge_types, dst_node_types)):
            # edge_name format is (src node type, edge type, dst node type)
            features = norm_src_features[i], norm_edge_feature_dict[i], norm_dst_features[i]
            if edge_name not in message_dict:
                message_dict[edge_name] = {i: features}
            else:
                message_dict[edge_name][i] = features


        # final message encoding
        messages = torch.zeros(len(src_features), self.emb_dim)
        for edge_name, feature_dict in message_dict.items():
            # edge_name format is (src node type, edge type, dst node type)
            idx = torch.tensor(list(feature_dict.keys()))
            src_f = torch.stack([i[0] for i in feature_dict.values()])
            edge_f = torch.stack([i[1] for i in feature_dict.values()])
            dst_f = torch.stack([i[2] for i in feature_dict.values()])
            concat_src = torch.cat([src_f, edge_f, entity_emb[idx], action_emb[idx]], dim=1)
            messages[idx] = self.edge_encoders[edge_name[1]](src=concat_src, dst=dst_f)

        return messages


class SumAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce='sum')


class ExampleMessageTransformer(nn.Module):
    """
        Implements a transformer block for message cross attention in order to aggregate messages to their destination
    """

    def __init__(self,
                 emb_dim: int,
                 memory_dim: int,
                 msg_dim: int,
                 time_dim: int,
                 bias: bool = True,
                 dropout: float = 0.2,
                 n_head: int = 1,
                 expansion_factor: int = 2,
                 ):
        super().__init__()
        self.src_linear = nn.Linear(memory_dim, emb_dim, bias=bias)
        self.msg_linear = nn.Linear(msg_dim, emb_dim, bias=bias)
        self.dst_linear = nn.Linear(memory_dim, emb_dim, bias=bias)

        self.src_norm = nn.LayerNorm(memory_dim, bias=bias)
        self.msg_norm = nn.LayerNorm(msg_dim, bias=bias)
        self.dst_norm = nn.LayerNorm(memory_dim, bias=bias)

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
        self.out_channels = emb_dim + time_dim

    def forward(self, src, dst, msg, t_enc):
        # norm and align
        src = self.src_linear(self.src_norm(src)) + self.msg_linear(self.msg_norm(msg))
        dst = self.dst_linear(self.dst_norm(dst))

        # Determine whether to compute attention weights based on the mode
        #need_weights = not self.training  # True if in eval mode, False if in training mode

        # Use src as both key and value in the attention mechanism
        attn_out, self.attn_weight = self.attn(query=dst.unsqueeze(0), key=src.unsqueeze(0), value=src.unsqueeze(0),
                                               is_causal=False, attn_mask=None
                                               )
        x = dst + attn_out.squeeze(0)  # Remove the batch dimension added for multi-head attention
        x = x + self.mlp(self.layer_norm(x))

        return torch.cat([x, t_enc], dim=-1)


class MLPMessageEncoder(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 memory_dim: int,
                 msg_dim: int,
                 time_dim: int,
                 bias: bool = True,
                 dropout: float = 0.2,
                 expansion_factor: int = 2,
                 ):
        super().__init__()
        self.src_norm = nn.LayerNorm(memory_dim, bias=bias)
        self.msg_norm = nn.LayerNorm(msg_dim, bias=bias)
        self.dst_norm = nn.LayerNorm(memory_dim, bias=bias)

        self.linear = nn.Linear(memory_dim * 2 + msg_dim, emb_dim, bias=bias)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * expansion_factor, bias=bias),
            nn.ReLU(),
            nn.Linear(emb_dim * expansion_factor, emb_dim, bias=bias),
            nn.Dropout(dropout)
        )

        self.out_channels = emb_dim + time_dim

    def forward(self, src, dst, msg, t_enc):
        # norm and align
        src = self.src_norm(src)
        msg = self.msg_norm(msg)
        dst = self.dst_norm(dst)
        x = torch.cat([src, dst, msg], dim=-1)
        x = self.linear(x)
        x = x + self.mlp(x)

        return torch.cat([x, t_enc], dim=-1)


class GraphAttention(nn.Module):
    def __init__(self,
                 memory_dim: int,
                 msg_dim: int,
                 time_dim: int,
                 bias: True
                 ):
        super(GraphAttention, self).__init__()

        self.src_norm = nn.LayerNorm(memory_dim, bias=bias)
        self.msg_norm = nn.LayerNorm(msg_dim, bias=bias)
        self.dst_norm = nn.LayerNorm(memory_dim, bias=bias)

        # Learnable parameters
        self.a_s = nn.Parameter(torch.randn(memory_dim, 1))
        self.a_t = nn.Parameter(torch.randn(memory_dim, 1))
        self.a_e = nn.Parameter(torch.randn(msg_dim, 1))
        self.leakyrelu = nn.LeakyReLU()

        self.out_channels = memory_dim + time_dim

    def forward(self, src, dst, msg, t_enc):

        dst = self.dst_norm(dst)

        # Calculate attention coefficients
        source_scores = self.leakyrelu(torch.matmul(self.src_norm(src), self.a_s))
        destination_scores = self.leakyrelu(torch.matmul(dst, self.a_t))
        msg_scores = self.leakyrelu(torch.matmul(self.msg_norm(msg), self.a_e))

        # Combine scores
        combined_scores = source_scores + destination_scores + msg_scores

        # Apply softmax to get attention weights
        attention_weights = F.softmax(combined_scores, dim=1)

        # Apply attention to destination features
        x = torch.mul(dst, attention_weights)

        return torch.cat([x, t_enc], dim=-1)


class AttentionMessageMemory(torch.nn.Module):
    def __init__(self, input_channels, emb_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_channels = input_channels
        self.emb_dim = emb_dim
        self.src_norm = nn.LayerNorm(input_channels, bias=True)
        self.src_linear = nn.Linear(input_channels, emb_dim, bias=True)

        self.hidden_norm = nn.LayerNorm(emb_dim, bias=True)

        self.attn = nn.MultiheadAttention(num_heads=1,
                                          embed_dim=emb_dim,
                                          bias=True,
                                          dropout=.1
                                          )
        self.layer_norm = nn.LayerNorm(emb_dim, bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2, bias=True),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim, bias=True),
        )

        self.attn_weight = None

    def forward(self, x, hidden_state):
        x = self.src_linear(self.src_norm(x))
        hidden_state = self.hidden_norm(hidden_state)
        # Use src as both key and value in the attention mechanism
        attn_out, self.attn_weight = self.attn(query=hidden_state.unsqueeze(0), key=x.unsqueeze(0),
                                               value=x.unsqueeze(0),
                                               is_causal=False, attn_mask=None
                                               )
        hidden_state = hidden_state + attn_out.squeeze(0)  # Remove the batch dimension added for multi-head attention
        return hidden_state + self.mlp(self.layer_norm(hidden_state))
