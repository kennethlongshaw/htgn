from itertools import islice
from src.nn.protocols import MemoryBatch
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, List, Optional, get_type_hints
from dataclasses import fields
from src.data_stores.stores import EdgeStore

def iter_index(iterable, value, start=0, stop=None):
    "Return indices where a value occurs in a sequence or iterable."
    # iter_index('AABCADEAF', 'A') → 0 1 4 7
    seq_index = getattr(iterable, 'index', None)
    if seq_index is None:
        # Path for general iterables
        it = islice(iterable, start, stop)
        for i, element in enumerate(it, start):
            if element is value or element == value:
                yield i
    else:
        # Path for sequences with an index() method
        stop = len(iterable) if stop is None else stop
        i = start
        try:
            while True:
                yield (i := seq_index(value, i, stop))
                i += 1
        except ValueError:
            pass


def filter_by_index(data: list, index: list, selection: int):
    mask = list(iter_index(iterable=index, value=selection))
    return [data[i] for i in mask]


def zdict_lookup(emb_dim: int,
                 mapping: dict[int: tuple[str, int]],
                 z_dict: dict[str: Tensor],
                 ids: Tensor):

    # Initialize the output tensor
    output = torch.zeros((len(ids), emb_dim), dtype=next(iter(z_dict.values())).dtype)

    for i, node_id in enumerate(ids):
        node_type, local_id = mapping[node_id.item()]
        output[i] = z_dict[node_type][local_id]

    return output


@torch.no_grad()
def batch_to_graph(batch: EdgeStore,
                   memories: Tensor,
                   memory_ids: Tensor
                   ) -> Tuple[HeteroData, Dict]:
    # Initialize the heterogeneous graph and global mapping
    graph = HeteroData()
    global_mapping = {}

    # Combine source and destination nodes, and get unique nodes
    nodes = torch.cat([torch.stack([batch.src_ids, batch.src_node_types], dim=1),
                       torch.stack([batch.dst_ids, batch.dst_node_types], dim=1)]).unique(dim=0)

    # Create a mapping from memory_ids to their indices
    mem_idx = {idx.item(): i for i, idx in enumerate(memory_ids)}

    # Process nodes for each type
    types_idx = {}
    for node_type in nodes[:, 1].unique():
        # Get nodes of this type
        type_nodes = nodes[nodes[:, 1] == node_type]

        # Create a mapping from node ID to local index for this type
        type_idx = {n[0].item(): i for i, n in enumerate(type_nodes)}

        # Update global mapping
        for k, v in type_idx.items():
            global_mapping[k] = (f'node_{node_type.item()}', v)

        # Get memory indices for nodes of this type
        mem_type_idx = [mem_idx[t] for t in type_idx.keys()]

        # Assign node features from memories
        graph[f'node_{node_type.item()}'].x = memories[mem_type_idx]

        # Store type index mapping
        types_idx[node_type.item()] = type_idx

    # Convert global node IDs to type-specific indices
    src_type_indices = torch.tensor(
        [types_idx[t.item()][s.item()] for s, t in zip(batch.src_ids, batch.src_node_types)])
    dst_type_indices = torch.tensor(
        [types_idx[t.item()][d.item()] for d, t in zip(batch.dst_ids, batch.dst_node_types)])

    # Add edges to the graph
    for edge_type in batch.edge_types.unique():
        edge_mask = batch.edge_types == edge_type
        src_edge_indices = src_type_indices[edge_mask]
        dst_edge_indices = dst_type_indices[edge_mask]
        src_type = batch.src_node_types[edge_mask][0].item()
        dst_type = batch.dst_node_types[edge_mask][0].item()
        edge_key = (f'node_{src_type}', f'edge_{edge_type.item()}', f'node_{dst_type}')
        graph[edge_key].edge_index = torch.stack([src_edge_indices, dst_edge_indices])

        # Add relative time encoding as edge attribute
        graph[edge_key].edge_attr = batch.rel_t_enc[edge_mask].unsqueeze(1)  # Ensure it's 2D

    return graph, global_mapping


def df_to_batch(df) -> MemoryBatch:
    non_feature_cols = [col for col in df.columns if 'feature' not in col]
    batch_data = df[non_feature_cols].to_torch(return_type='dict')
    feature_cols = [col for col in df.columns if 'feature' in col]
    features = {col: df[col].tolist() for col in feature_cols}
    batch_data.update(features)
    return MemoryBatch(**batch_data)


def concat_memory_batches(batches: List[MemoryBatch]) -> MemoryBatch:
    def concat_or_none(tensors: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
        tensors = [t for t in tensors if t is not None]
        return torch.cat(tensors) if tensors else None

    def concat_features(features_list: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        return [torch.cat(features) for features in zip(*features_list)]

    concatenated_data = {}
    type_hints = get_type_hints(MemoryBatch)

    for field in fields(MemoryBatch):
        field_name = field.name
        field_type = type_hints[field_name]

        values = [getattr(batch, field_name) for batch in batches]

        if field_type == List[torch.Tensor]:
            concatenated_data[field_name] = concat_features(values)
        elif field_type == Optional[torch.Tensor]:
            if any(isinstance(v, list) for v in values):  # Check if any value is a list
                concatenated_data[field_name] = concat_features([v for v in values if v is not None])
            else:
                concatenated_data[field_name] = concat_or_none(values)
        elif field_type == torch.Tensor:
            concatenated_data[field_name] = torch.cat(values)
        elif field_type == Optional[List[torch.Tensor]]:
            concatenated_data[field_name] = concat_features([v for v in values if v is not None])
        else:
            concatenated_data[field_name] = values[0]

    return MemoryBatch(**concatenated_data)
