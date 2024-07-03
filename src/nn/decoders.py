import torch.nn as nn
import torch
from src.nn.reference_transformers import Transformer


class DotProductLinkPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_dict: dict, edge_label_indices: dict, edge_types: list) -> dict:
        """
        Decodes edges for multiple edge types by calculating the dot product between embeddings of node pairs
        and then applying a sigmoid function to map the scores to probabilities.

        Args:
            z_dict (dict): Dictionary of node embeddings, keyed by node type.
            edge_label_indices (dict): A dictionary where keys are edge types (tuples) and values are the indices
                                       of the edges to be decoded, each value having the shape [2, num_edges] for that edge type.
            edge_types (list): A list of tuples representing multiple edge types in the form
                               (source_node_type, relation_type, target_node_type).

        Returns:
            dict: A dictionary where each key is an edge type and the value is a tensor of probabilities
                  for each edge, indicating the likelihood of edge existence.
        """
        edge_probabilities = {}

        for edge_type in edge_types:
            source_node_type, _, target_node_type = edge_type
            edge_label_index = edge_label_indices[edge_type]

            src_indices, tgt_indices = edge_label_index

            src_embeddings = z_dict[source_node_type][src_indices]
            tgt_embeddings = z_dict[target_node_type][tgt_indices]

            edge_scores = (src_embeddings * tgt_embeddings).sum(dim=1)
            edge_probabilities[edge_type] = torch.sigmoid(edge_scores)

        return edge_probabilities


class LinearLinkPredictor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = nn.Linear(in_channels, in_channels)
        self.lin_dst = nn.Linear(in_channels, in_channels)
        self.lin_final = nn.Linear(in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z_dict, edge_label_indices, edge_types):
        """
        Decodes edges for multiple types using linear transformations followed by ReLU and sigmoid activation.

        Args:
            z_dict (dict): Node embeddings keyed by node type.
            edge_label_indices (dict): Indices of edges to decode, keyed by edge type.
            edge_types (list): Edge types as (source_node_type, _, target_node_type).

        Returns:
            dict: Probabilities of edge existence keyed by edge type.
        """
        edge_probabilities = {}
        for et in edge_types:
            src_embeddings = z_dict[et[0]][edge_label_indices[et][0]]
            dst_embeddings = z_dict[et[2]][edge_label_indices[et][1]]

            # Apply linear transformations and sum
            h = self.lin_src(src_embeddings) + self.lin_dst(dst_embeddings)
            h = torch.relu(h)  # Apply ReLU activation

            # Final linear transformation and apply sigmoid
            edge_probabilities[et] = self.sigmoid(self.lin_final(h))

        return edge_probabilities


class LinearPredictor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = nn.Linear(in_channels, in_channels)
        self.lin_dst = nn.Linear(in_channels, in_channels)
        self.lin_final = nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


class MultiLinearLinkPredictor(nn.Module):
    def __init__(self,
                 edge_types,
                 in_channels):
        super().__init__()
        self.edge_layers = torch.nn.ModuleDict({
            str(et): LinearLinkPredictor(in_channels=in_channels)
            for et in edge_types
        })

    def forward(self, z_dict, edge_label_indices, edge_types):
        """
        Creates a Linear decoder for each edge type

        Args:
            z_dict (dict): Node embeddings keyed by node type.
            edge_label_indices (dict): Indices of edges to decode, keyed by edge type.
            edge_types (list): Edge types as (source_node_type, _, target_node_type).

        Returns:
            dict: Probabilities of edge existence keyed by edge type.
        """
        edge_probabilities = {}
        for et in edge_types:
            edge_layer = self.edge_layers[et]
            src_embeddings = z_dict[et[0]][edge_label_indices[et][0]]
            dst_embeddings = z_dict[et[2]][edge_label_indices[et][1]]

            # Final linear transformation and apply sigmoid
            edge_probabilities[et] = edge_layer(src_embeddings, dst_embeddings)

        return edge_probabilities


class BilinearLinkPredictor(nn.Module):
    def __init__(self, in_channels, edge_types):
        super().__init__()
        self.edge_types = edge_types
        self.bilinear_layers = torch.nn.ModuleDict({
            str(et): nn.Linear(in_channels, in_channels, bias=False)
            for et in edge_types
        })

    def forward(self, z_dict, edge_label_indices):
        """
        Decodes edges for multiple types using bilinear transformations.

        Args:
            z_dict (dict): Node embeddings keyed by node type.
            edge_label_indices (dict): Indices of edges to decode, keyed by edge type.

        Returns:
            dict: Probabilities of edge existence keyed by edge type.
        """
        edge_probabilities = {}
        for et in self.edge_types:
            src_type, _, dst_type = et
            src_indices, dst_indices = edge_label_indices[et]

            src_embeddings = z_dict[src_type][src_indices]
            dst_embeddings = z_dict[dst_type][dst_indices]

            # Apply bilinear transformation via linear layer without bias
            transformed_src = self.bilinear_layers[str(et)](src_embeddings)

            # Compute the interaction scores using element-wise multiplication and sum
            interaction_scores = (transformed_src * dst_embeddings).sum(dim=1)

            # Apply sigmoid to get probabilities
            edge_probabilities[et] = torch.sigmoid(interaction_scores)

        return edge_probabilities


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

        self.transformers = nn.ModuleDict({
            et: nn.Sequential(Transformer(emb_dim=emb_dim,
                                          dropout=dropout,
                                          expansion_factor=mlp_expansion,
                                          num_heads=num_heads,
                                          bias=bias
                                          ),
                              nn.Linear(emb_dim, 1),
                              nn.Sigmoid()
                              )
            for et in edge_types
        })

    def forward(self, z_dict, edge_label_indices):
        """
        Applies a transformer-based attention mechanism and a linear layer to predict link probabilities for multiple edge types.

        Args:
            z_dict (dict): Node embeddings keyed by node type.
            edge_label_indices (dict): Indices of edges to decode, keyed by edge type.

        Returns:
            dict: Probabilities of edge existence keyed by edge type.
        """
        edge_probabilities = {}
        for et in self.edge_types:
            src_type, _, dst_type = et
            src_indices, dst_indices = edge_label_indices[et]

            # Get embeddings for source and destination nodes
            z_src = z_dict[src_type][src_indices]
            z_dst = z_dict[dst_type][dst_indices]

            # Process using the attention module specific to the edge type
            attn_output, _ = self.attentions[et](query=z_src, key=z_dst, value=z_dst)

            # Apply the linear transformation and sigmoid activation for each sequence element
            edge_probabilities[et] = self.mlp_block[et](attn_output)

        return edge_probabilities
