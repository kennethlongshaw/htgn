from src.nn.protocols import MemoryBatch
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from src.data_stores.stores import EdgeStore
from typing import Callable, Dict


def zdict_lookup(hgraph: HeteroData,
                 z_dict: Dict[str, torch.Tensor],
                 ids: torch.Tensor) -> torch.Tensor:
    embeddings = []
    for node_type in hgraph.node_types:
        # Get the node_ids for this node type
        node_ids = hgraph[node_type].node_id

        # Find which of the lookup ids are of this node type
        mask = torch.isin(ids, node_ids)
        type_ids = ids[mask]

        if len(type_ids) > 0:
            # Get the local indices of these ids within this node type
            _, type_indices = torch.where(node_ids.unsqueeze(1) == type_ids.unsqueeze(0))

            # Lookup the embeddings
            type_embeddings = z_dict[node_type][type_indices]
            embeddings.append(type_embeddings)

    return torch.cat(embeddings, dim=0)


def add_graph_features(hgraph: HeteroData,
                       memory_ids: Tensor,
                       memories: Tensor,
                       time_enc: Callable
                       ) -> HeteroData:
    # Create a mapping from memory_ids to memories
    memory_dict = dict(zip(memory_ids.tolist(), memories))

    for node in hgraph.metadata()[0]:
        node_ids = hgraph[node].node_id
        # Directly assign memories to nodes using the mapping
        hgraph[node].x = torch.cat([memory_dict[n_id.item()] for n_id in node_ids])

    for edge in hgraph.metadata()[1]:
        hgraph[edge].edge_attr = time_enc(hgraph[edge].rel_time)

    return hgraph


@torch.no_grad()
def batch_to_graph(batch: EdgeStore,
                   ) -> HeteroData:
    graph = HeteroData()

    # Combine all nodes and get unique nodes with their types
    all_nodes = torch.cat([batch.src_ids, batch.dst_ids])
    all_node_types = torch.cat([batch.src_node_types, batch.dst_node_types])
    unique_nodes, inverse_indices = torch.unique(torch.stack([all_nodes, all_node_types], dim=1), dim=0,
                                                 return_inverse=True)

    # Create node type masks
    node_type_masks = {nt.item(): unique_nodes[:, 1] == nt for nt in torch.unique(all_node_types)}

    # Add node features
    for node_type, mask in node_type_masks.items():
        type_nodes = unique_nodes[mask, 0]
        graph[f'node_{node_type}'].node_ids = type_nodes

    # Process edges
    edge_types = batch.edge_types.unique()
    for edge_type in edge_types:
        mask = batch.edge_types == edge_type
        src_type = batch.src_node_types[mask][0].item()
        dst_type = batch.dst_node_types[mask][0].item()

        src_indices = inverse_indices[:len(batch.src_ids)][mask]
        dst_indices = inverse_indices[len(batch.src_ids):][mask]

        edge_key = (f'node_{src_type}', f'edge_{edge_type.item()}', f'node_{dst_type}')
        graph[edge_key].edge_index = torch.stack([src_indices, dst_indices])
        graph[edge_key].rel_time = batch.rel_time[mask]

    return graph


def df_to_batch(df) -> MemoryBatch:
    non_feature_cols = [col for col in df.columns if 'feature' not in col]
    feature_cols = [col for col in df.columns if 'feature' in col]
    features = {col: df[col].to_list() for col in feature_cols}

    batch_data = df[non_feature_cols].to_torch(return_type='dict')
    batch_data.update(features)
    batch = MemoryBatch(**batch_data)

    batch.src_features = [torch.tensor(f) for f in batch.src_features]
    batch.dst_features = [torch.tensor(f) for f in batch.dst_features]
    batch.edge_features = [torch.tensor(f) for f in batch.edge_features]
    return batch
