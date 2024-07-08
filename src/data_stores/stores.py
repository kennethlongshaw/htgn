import torch.nn as nn
import torch
from torch import Tensor
import src.utils.utils as utils
from torch_geometric.utils import scatter
from torch_geometric.nn.inits import zeros
from src.nn.protocols import MemoryBatch
from typing import List, get_type_hints
from dataclasses import dataclass, fields


@dataclass
class EdgeStore:
    time: Tensor = None
    rel_t: Tensor = None
    rel_t_enc: Tensor = None
    src_ids: Tensor = None
    src_node_types: Tensor = None
    dst_ids: Tensor = None
    dst_node_types: Tensor = None
    edge_types: Tensor = None
    edge_features: list[Tensor] = None

    def to(self, device: torch.device) -> 'EdgeStore':
        """Move the store data to a specified device."""
        new_kwargs = {}
        type_hints = get_type_hints(self.__class__)
        for field in fields(self):
            value = getattr(self, field.name)
            field_type = type_hints[field.name]
            if value is None:
                new_kwargs[field.name] = None
            elif isinstance(field_type, list):
                new_kwargs[field.name] = [v.to(device) for v in value]
            elif isinstance(field_type, torch.Tensor):
                new_kwargs[field.name] = value.to(device)
            else:
                new_kwargs[field.name] = value
        return EdgeStore(**new_kwargs)

    def get_edges(self, filter_ids: Tensor, filter_target='dst'):
        # If the store is empty or the target field is None, return an empty EdgeStore
        if getattr(self, f"{filter_target}_ids") is None:
            return EdgeStore()

        type_hints = get_type_hints(self.__class__)
        filter_masks = {
            'src': lambda filter_id: self.src_ids == filter_id,
            'dst': lambda filter_id: self.dst_ids == filter_id,
        }
        if filter_target not in filter_masks:
            raise Exception(f"Invalid filter target: {filter_target}")

        # Ensure filter_ids is a tensor
        if not isinstance(filter_ids, torch.Tensor):
            filter_ids = torch.tensor(filter_ids, device=getattr(self, f"{filter_target}_ids").device)

        mask = torch.stack([filter_masks[filter_target](filter_id) for filter_id in filter_ids]).any(dim=0)

        new_kwargs = {}
        for field in fields(self):
            value = getattr(self, field.name)
            field_type = type_hints[field.name]
            if value is None:
                new_kwargs[field.name] = None
            elif isinstance(value, list) or field_type == List[torch.Tensor]:
                new_kwargs[field.name] = [v[mask] for v in value] if value else []
            else:
                new_kwargs[field.name] = value[mask] if value is not None else None

        return EdgeStore(**new_kwargs)

    def append(self, batch: MemoryBatch) -> 'EdgeStore':
        new_time = batch.time
        new_src_ids = batch.src_ids
        new_dst_ids = batch.dst_ids
        new_edge_types = batch.edge_types
        new_edge_features = batch.edge_features

        new_kwargs = {}
        for field in fields(self):
            current_value = getattr(self, field.name)
            new_value = locals()[f"new_{field.name}"]

            if isinstance(current_value, torch.Tensor):
                new_kwargs[field.name] = torch.cat([current_value, new_value])
            elif isinstance(current_value, list):
                new_kwargs[field.name] = [torch.cat([curr, new]) for curr, new in zip(current_value, new_value)]
            else:
                raise ValueError(f"Unsupported field type for {field.name}")

        return EdgeStore(**new_kwargs)


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
    def set_memory(self, dst_ids: Tensor, memory: Tensor):
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
        return utils.concat_memory_batches(msgs)

    def reset_state(self):
        self.msg_store = {}


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
    def calc_relative_time(self,
                           dst_ids: Tensor,
                           times: Tensor):
        last_update = self.last_update[dst_ids]
        unique, index = dst_ids.unique(return_inverse=True)
        return last_update[index] - times

    def reset_state(self):
        zeros(self.last_update)
