from itertools import islice
#from ..nn.memory_module import MemoryBatch
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from dataclasses import dataclass

@dataclass
class MemoryBatch:
    rel_time: Tensor
    entity_types: Tensor
    action_types: Tensor

    src_node_types: Tensor
    src_features: list[Tensor]
    src_memories: Tensor

    edge_types: Tensor
    edge_features: list[Tensor]
    edge_labels: Tensor

    dst_node_types: Tensor
    dst_features: list[Tensor]
    dst_memories: Tensor


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


@torch.no_grad()
# def batch_to_graph(batch: MemoryBatch,
#                    src_ids: Tensor,
#                    dst_ids: Tensor,
#                    memories: Tensor,
#                    memory_ids: Tensor
#                    ) -> HeteroData:
#     graph = HeteroData()
#     nodes = torch.cat([torch.stack([src_ids, batch.src_node_types], dim=1),
#                        torch.stack([dst_ids, batch.dst_node_types], dim=1)]).unique(dim=0)
#     # Assign node features from memories, indexed by memory_ids
#     mem_idx = {idx.item(): i for i, idx in enumerate(memory_ids)}
#     types_idx = {}
#     for node_type in nodes[:, 1].unique():
#         type_nodes = nodes[nodes[:, 1] == node_type]
#         type_idx = {n[0].item(): i for i, n in enumerate(type_nodes)}
#         mem_type_idx = [mem_idx[t] for t in type_idx.keys()]
#         graph[f'node_{node_type.item()}'].x = memories[mem_type_idx]
#         types_idx[node_type.item()] = type_idx
#
#     # Add edges to the graph
#     src_type_indices = torch.tensor([types_idx[t.item()][s.item()] for s, t in zip(src_ids, batch.src_node_types)])
#     dst_type_indices = torch.tensor([types_idx[t.item()][d.item()] for d, t in zip(dst_ids, batch.dst_node_types)])
#
#     for edge_type in batch.edge_types.unique():
#         edge_mask = batch.edge_types == edge_type
#         src_edge_indices = src_type_indices[edge_mask]
#         dst_edge_indices = dst_type_indices[edge_mask]
#         src_type = batch.src_node_types[edge_mask][0].item()
#         dst_type = batch.dst_node_types[edge_mask][0].item()
#         graph[f'node_{src_type}', f'edge_{edge_type.item()}', f'node_{dst_type}'].edge_index = torch.stack(
#             [src_edge_indices, dst_edge_indices])
#
#     return graph

@torch.no_grad()
def batch_to_graph(batch: MemoryBatch,
                   src_ids: Tensor,
                   dst_ids: Tensor,
                   memories: Tensor,
                   memory_ids: Tensor
                   ) -> HeteroData:
    graph = HeteroData()
    nodes = torch.cat([torch.stack([src_ids, batch.src_node_types], dim=1),
                       torch.stack([dst_ids, batch.dst_node_types], dim=1)]).unique(dim=0)
    # Assign node features from memories, indexed by memory_ids
    mem_idx = {idx.item(): i for i, idx in enumerate(memory_ids)}
    types_idx = {}
    for node_type in nodes[:, 1].unique():
        type_nodes = nodes[nodes[:, 1] == node_type]
        type_idx = {n[0].item(): i for i, n in enumerate(type_nodes)}
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
        graph[edge_key].edge_label = batch.edge_labels[edge_mask]  # Add edge labels
    return graph