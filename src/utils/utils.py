from itertools import islice
from src.nn.protocols import MemoryBatch
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, List, Optional, get_type_hints
from dataclasses import fields

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


def zdict_lookup(hgraph: HeteroData,
                 z_dict: dict,
                 ids: Tensor):
    "z dict will have embeddings for nodes by type in the order they were in according to hgraph"
    "hgraph will have the node id"
    "ids will have the ids in the order we want"

    # Must return in same order as given
    return


@torch.no_grad()
def batch_to_graph(batch: MemoryBatch,
                   src_ids: Tensor,
                   dst_ids: Tensor,
                   memories: Tensor,
                   memory_ids: Tensor
                   ) -> Tuple[HeteroData, Dict]:
    """
    Convert a MemoryBatch object to a HeteroData graph object.

    Args:
        batch (MemoryBatch): The MemoryBatch object containing the batch data.
        src_ids (Tensor): The source node IDs.
        dst_ids (Tensor): The destination node IDs.
        memories (Tensor): The memory tensor containing node features.
        memory_ids (Tensor): The memory IDs corresponding to the node IDs.

    Returns:
        HeteroData: The converted HeteroData graph object.
        Global Mapping: The local node IDs by type and they're global IDs as values

    Description:
        This function takes a MemoryBatch object and converts it into a HeteroData graph object.
        The graph object contains nodes and edges with their corresponding features and labels.
        The node features are assigned from the memories tensor, indexed by the memory_ids.
        The edges are added to the graph based on the source and destination node IDs and types.

        The resulting HeteroData graph object has the following structure:
        - Nodes: Nodes are identified by their type and ID. The node features are stored in the
                 'x' attribute of each node type.
        - Edges: Edges are identified by their source node type, edge type, and destination node type.
                 The edge indices are stored in the 'edge_index' attribute of each edge type.


    Note:
        The function assumes that the MemoryBatch object contains the necessary attributes such as
        src_node_types, dst_node_types, edge_types, and record_id.
        The memory_ids tensor should have the same length as the unique node IDs in the batch.
        The src_ids and dst_ids tensors should have the same length as the number of edges in the batch.
    """

    graph = HeteroData()
    global_mapping = {}
    nodes = torch.cat([torch.stack([src_ids, batch.src_node_types], dim=1),
                       torch.stack([dst_ids, batch.dst_node_types], dim=1)]).unique(dim=0)

    # Assign node features from memories, indexed by memory_ids
    mem_idx = {idx.item(): i for i, idx in enumerate(memory_ids)}
    types_idx = {}
    for node_type in nodes[:, 1].unique():
        type_nodes = nodes[nodes[:, 1] == node_type]
        type_idx = {n[0].item(): i for i, n in enumerate(type_nodes)}

        for k, v in type_idx.items():
            global_mapping[k] = (f'node_{node_type.item()}', v)

        mem_type_idx = [mem_idx[t] for t in type_idx.keys()]
        graph[f'node_{node_type.item()}'].x = memories[mem_type_idx]

        types_idx[node_type.item()] = type_idx

    # Add edges to the graph
    src_type_indices = torch.tensor([types_idx[t.item()][s.item()] for s, t in zip(src_ids, batch.src_node_types)])
    dst_type_indices = torch.tensor([types_idx[t.item()][d.item()] for d, t in zip(dst_ids, batch.dst_node_types)])
    for edge_type in batch.edge_types.unique():
        edge_mask = batch.edge_types == edge_type
        src_edge_indices = src_type_indices[edge_mask]
        dst_edge_indices = dst_type_indices[edge_mask]
        src_type = batch.src_node_types[edge_mask][0].item()
        dst_type = batch.dst_node_types[edge_mask][0].item()
        edge_key = (f'node_{src_type}', f'edge_{edge_type.item()}', f'node_{dst_type}')
        graph[edge_key].edge_index = torch.stack([src_edge_indices, dst_edge_indices])

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