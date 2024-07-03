import torch.nn as nn
import polars as pl
import torch
from torch import Tensor
from src.utils.utils import df_to_batch, concat_memory_batches
from torch_geometric.utils import scatter
from torch_geometric.nn.inits import zeros
from src.nn.protocols import MemoryBatch
from typing import List

from collections import namedtuple



class EdgeStore(nn.Module):
    def __init__(self,
                 ):
        super().__init__()
        self.edge_store = None
        self.edge_features = []

    def insert_edges(self,
                     times: Tensor,
                     src_ids: Tensor,
                     edge_types: Tensor,
                     edge_features: List[Tensor],
                     dst_ids: Tensor):
        edges = torch.stack([times, src_ids, edge_types, dst_ids])
        if self.edge_store is None:
            self.edge_store = edges
        else:
            self.edge_store = torch.cat([self.edge_store, edges])
            self.edge_features += edge_features

    def get_neighbors(self, node_ids: Tensor) -> tuple[Tensor, List[Tensor]]:
        if self.edge_store is None:
            return None, []

        # Find indices where dst_ids match
        mask = torch.isin(self.edge_store[3], node_ids)

        # Select matching edges
        selected_edges = self.edge_store[:, mask]
        selected_features = [feature[mask] for feature in self.edge_features]

        return selected_edges, selected_features


class MemoryStore(nn.Module):
    def __init__(self,
                 num_nodes,
                 memory_dim
                 ):
        super().__init__()
        self.register_buffer('memory_store', torch.zeros(num_nodes, memory_dim))

    @torch.no_grad()
    def get_memory(self, dst_ids: Tensor):
        return self.memory_store[dst_ids]

    @torch.no_grad()
    def set_memory(self, dst_ids, memory):
        self.memory_store[dst_ids] = memory

    def reset_state(self):
        zeros(self.memory_store)


class MessageStore(nn.Module):
    def __init__(self):
        super().__init__()
        self.msg_store = {}

    @torch.no_grad()
    def set_msg_store(self, batch: MemoryBatch) -> None:
        unique, index = batch.dst_ids.unique()
        for dst_id in unique:
            self.msg_store[dst_id] = batch.filter_by_ids([dst_id])

    @torch.no_grad()
    def get_from_msg_store(self, dst_ids: Tensor) -> MemoryBatch:
        msgs = [self.msg_store.get(i, None) for i in dst_ids.tolist() if self.msg_store.get(i, None) is not None]
        return concat_memory_batches(msgs)


class LastUpdateStore(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 ):
        super().__init__()
        self.num_nodes = num_nodes
        self.register_buffer('last_update', torch.empty(num_nodes, dtype=torch.long))

    @torch.no_grad()
    def get_last_update(self, dst_ids: Tensor):
        return self.last_update[dst_ids]

    @torch.no_grad()
    def set_last_update(self, dst_ids: Tensor, times: Tensor):
        unique, index = dst_ids.unique(return_inverse=True)
        new_update = scatter(src=times, index=index, dim_size=self.num_nodes, reduce='max')
        self.last_update = torch.maximum(self.last_update, new_update)

    @torch.no_grad()
    def calc_relative_time(self, dst_ids: Tensor, times: Tensor):
        last_update = self.last_update[dst_ids]
        unique, index = dst_ids.unique(return_inverse=True)
        return last_update[index] - times

    def reset_state(self):
        zeros(self.last_update)
