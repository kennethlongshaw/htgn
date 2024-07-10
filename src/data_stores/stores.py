import torch.nn as nn
import torch
from torch import Tensor
import src.utils.utils as utils
from torch_geometric.utils import scatter
from torch_geometric.nn.inits import zeros
from src.nn.protocols import MemoryBatch
from typing import List, get_type_hints, get_origin, get_args, Union
from dataclasses import dataclass, fields


@dataclass
class EdgeStore:
    time: Tensor = None
    rel_time: Tensor = None
    rel_time_enc: Tensor = None
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

    def get_edges(self, nodes: torch.Tensor) -> 'EdgeStore':
        mask = torch.isin(self.src_ids, nodes) | torch.isin(self.dst_ids, nodes)
        new_kwargs = {}

        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                if value.shape[0] == mask.shape[0]:
                    new_kwargs[field.name] = value[mask]
                else:
                    print(
                        f"Warning: Shape mismatch for field {field.name}. Mask shape: {mask.shape}, Tensor shape: {value.shape}")
                    new_kwargs[field.name] = value  # Keep the original tensor if shapes don't match
            elif isinstance(value, list):
                if value:
                    new_kwargs[field.name] = [v[mask[:v.shape[0]]] if v.shape[0] == mask.shape[0] else v for v in value]
                else:
                    new_kwargs[field.name] = []
            else:
                new_kwargs[field.name] = value

        return EdgeStore(**new_kwargs)

    def append(self, batch: MemoryBatch) -> 'EdgeStore':
        new_time = batch.time
        new_src_ids = batch.src_ids
        new_src_node_types = batch.src_node_types
        new_dst_ids = batch.dst_ids
        new_dst_node_types = batch.dst_node_types
        new_edge_types = batch.edge_types
        new_edge_features = batch.edge_features
        new_rel_time = batch.rel_time
        new_rel_time_enc = batch.rel_time_enc
        new_kwargs = {}

        type_hints = get_type_hints(self.__class__)

        for field in fields(self):
            current_value = getattr(self, field.name)
            new_value = locals()[f"new_{field.name}"]
            expected_type = type_hints[field.name]

            if current_value is None:
                if self._check_type(new_value, expected_type):
                    new_kwargs[field.name] = new_value
                else:
                    raise TypeError(
                        f"New value for {field.name} does not match the expected type. Expected {expected_type}, got {type(new_value)}")
            elif self._is_tensor_type(expected_type):
                if isinstance(current_value, torch.Tensor) and isinstance(new_value, torch.Tensor):
                    new_kwargs[field.name] = torch.cat([current_value, new_value])
                else:
                    raise TypeError(f"Both current and new values for {field.name} must be tensors")
            elif self._is_list_of_tensors_type(expected_type):
                if isinstance(current_value, list) and isinstance(new_value, list):
                    if all(isinstance(x, torch.Tensor) for x in current_value + new_value):
                        new_kwargs[field.name] = [torch.cat([curr, new]) for curr, new in zip(current_value, new_value)]
                    else:
                        raise TypeError(f"All elements in the list for {field.name} must be tensors")
                else:
                    raise TypeError(f"Both current and new values for {field.name} must be lists")
            else:
                raise ValueError(f"Unsupported field type for {field.name}. Expected: {expected_type}")

        return EdgeStore(**new_kwargs)

    def _check_type(self, value, expected_type):
        origin = get_origin(expected_type)
        if origin is Union:
            return any(self._check_type(value, t) for t in get_args(expected_type))
        elif origin is List:
            return isinstance(value, list) and all(self._check_type(item, get_args(expected_type)[0]) for item in value)
        elif origin is None:  # This means it's not a generic type
            return isinstance(value, expected_type)
        else:
            # For other generic types, we'll just check if it's an instance of the origin
            return isinstance(value, origin)

    def _is_tensor_type(self, t):
        return t is torch.Tensor or (get_origin(t) is Union and torch.Tensor in get_args(t))

    def _is_list_of_tensors_type(self, t):
        return (get_origin(t) is List and
                (get_args(t)[0] is torch.Tensor or
                 (get_origin(get_args(t)[0]) is Union and torch.Tensor in get_args(get_args(t)[0]))))


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
        for dst_id in batch.dst_ids.unique():
            self.msg_store[dst_id.item()] = batch.filter_by_ids([dst_id])

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
        self.register_buffer('last_update', torch.empty(num_nodes, dtype=torch.float))

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
