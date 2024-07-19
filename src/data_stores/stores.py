import torch.nn as nn
import torch
from torch import Tensor
from torch_geometric.utils import scatter
from src.nn.protocols import MemoryBatch
from dataclasses import dataclass
from itertools import compress


@dataclass
class EdgeStore:
    time: Tensor = None
    src_ids: Tensor = None
    src_node_types: Tensor = None
    dst_ids: Tensor = None
    dst_node_types: Tensor = None
    edge_types: Tensor = None
    edge_features: list[Tensor] = None
    rel_time: Tensor = None
    rel_time_enc: Tensor = None

    def to(self, device: torch.device) -> 'EdgeStore':
        self.time.to(device)

        self.src_ids.to(device)
        self.src_node_types.to(device)

        self.dst_ids.to(device)
        self.dst_node_types.to(device)

        self.edge_types.to(device)
        [ef.to(device) for ef in self.edge_features]

        if self.rel_time is not None:
            self.rel_time.to(device)
        if self.rel_time_enc is not None:
            self.rel_time_enc.to(device)

        return self

    def get_edges(self, nodes: torch.Tensor) -> 'EdgeStore':
        mask = torch.isin(self.dst_ids, nodes)
        return EdgeStore(time=self.time[mask],
                         src_ids=self.src_ids[mask],
                         src_node_types=self.src_node_types[mask],
                         dst_ids=self.dst_ids[mask],
                         dst_node_types=self.dst_node_types[mask],
                         edge_types=self.edge_types[mask],
                         edge_features=list(compress(self.edge_features, mask.tolist()))
                         )

    def append(self, batch: MemoryBatch) -> 'EdgeStore':
        def concat_or_new(existing: Tensor, new: Tensor) -> Tensor:
            return torch.cat([existing, new]) if existing is not None else new

        return EdgeStore(
            time=concat_or_new(self.time, batch.time),
            src_ids=concat_or_new(self.src_ids, batch.src_ids),
            src_node_types=concat_or_new(self.src_node_types, batch.src_node_types),
            dst_ids=concat_or_new(self.dst_ids, batch.dst_ids),
            dst_node_types=concat_or_new(self.dst_node_types, batch.dst_node_types),
            edge_types=concat_or_new(self.edge_types, batch.edge_types),
            edge_features=self.edge_features + batch.edge_features
            if self.edge_features is not None else batch.edge_features
        )

    def __add__(self, other):
        return self.append(other)


class MemoryStore(nn.Module):
    def __init__(self,
                 num_nodes,
                 memory_dim
                 ):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.register_buffer('memory_store', torch.zeros(num_nodes, memory_dim))

    @torch.no_grad()
    def get_memory(self, dst_ids: Tensor):
        return self.memory_store[dst_ids]

    @torch.no_grad()
    def set_memory(self, dst_ids: Tensor, memory: Tensor):
        self.memory_store[dst_ids] = memory

    def reset_state(self):
        self.register_buffer('memory_store', torch.zeros(self.num_nodes, self.memory_dim))


class MessageStore(nn.Module):
    def __init__(self):
        super().__init__()
        self.msg_store = {}

    @torch.no_grad()
    def set_msg_store(self, batch: MemoryBatch) -> None:
        for dst_id in batch.dst_ids.unique():
            self.msg_store[dst_id.item()] = batch.filter_by_ids([dst_id])

    @torch.no_grad()
    def get_from_msg_store(self, dst_ids: Tensor) -> MemoryBatch:
        msgs = [self.msg_store.get(i, None) for i in dst_ids.tolist() if self.msg_store.get(i, None) is not None]
        # memory batches are overloaded on __add__ for concat via sum
        return sum(msgs[1:], start=msgs[0])

    def reset_state(self):
        self.msg_store = {}


class LastUpdateStore(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 ):
        super().__init__()
        self.num_nodes = num_nodes
        self.register_buffer('last_update', torch.full((num_nodes,), float('-inf'), dtype=torch.float))

    @torch.no_grad()
    def get_last_update(self, dst_ids: Tensor):
        return self.last_update[dst_ids]

    @torch.no_grad()
    def set_last_update(self, dst_ids: Tensor, times: Tensor):
        new_update = scatter(src=times, index=dst_ids, dim_size=self.num_nodes, reduce='max')
        self.last_update = torch.maximum(self.last_update, new_update)

    @torch.no_grad()
    def calc_relative_time(self, dst_ids: Tensor, times: Tensor):
        last_update = self.last_update[dst_ids]
        # Replace -inf with current time for previously unseen nodes
        return times - torch.where(torch.isinf(last_update), times, last_update)

    def reset_state(self):
        self.register_buffer('last_update', torch.full((self.num_nodes,), float('-inf'), dtype=torch.float))
