from torch import Tensor, Optional
import torch
from typing import Protocol, Tuple, get_type_hints, List
from dataclasses import dataclass, fields
from torch_geometric.data import HeteroData
from itertools import compress

@dataclass(slots=True)
class MemoryBatch:
    entity_types: Tensor
    src_ids: Tensor
    src_node_types: Tensor
    src_features: list[Tensor]

    edge_types: Tensor

    dst_ids: Tensor
    dst_node_types: Tensor
    dst_features: list[Tensor]

    neg_ids: Optional[Tensor] = None
    action_types: Optional[Tensor] = None
    edge_features: Optional[Tensor] = None
    src_memories: Optional[Tensor] = None
    dst_memories: Optional[Tensor] = None
    time: Optional[Tensor] = None
    rel_time: Optional[Tensor] = None
    rel_time_enc: Optional[Tensor] = None

    def to(self, device: torch.device) -> 'MemoryBatch':
        """Move the batch to a specified device."""
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
        return MemoryBatch(**new_kwargs)

    def filter_by_ids(self, filter_ids, filter_target='dst'):
        type_hints = get_type_hints(self.__class__)
        filter_masks = {
            'src': lambda filter_id: self.src_ids == filter_id,
            'dst': lambda filter_id: self.dst_ids == filter_id,
            'neg': lambda filter_id: self.neg_ids == filter_id if self.neg_ids is not None else None
        }
        if filter_target not in filter_masks:
            raise Exception(f"Invalid filter target: {filter_target}")

        mask = torch.stack([filter_masks[filter_target](filter_id) for filter_id in filter_ids]).any(dim=0)

        new_kwargs = {}
        for field in fields(self):
            value = getattr(self, field.name)
            field_type = type_hints[field.name]
            if value is None:
                new_kwargs[field.name] = None
            elif isinstance(value, list) or field_type == List[torch.Tensor]:
                new_kwargs[field.name] = [v for v, m in zip(value, mask.tolist()) if m]
            else:
                new_kwargs[field.name] = value[mask]

        return MemoryBatch(**new_kwargs)


class MessageEncoderProtocol(Protocol):
    def __call__(self,
                 batch: MemoryBatch
                 ) -> Tensor:
        pass


class AggregatorProtocol(Protocol):
    def __call__(self,
                 index,
                 values,
                 num_nodes
                 ) -> Tensor:
        pass


class TimeEncoderProtocol(Protocol):
    def __call__(self,
                 t: Tensor
                 ) -> Tensor:
        pass


class MemoryProtocol(Protocol):
    def __call__(self,
                 input_tensor: Tensor,
                 hidden_state: Tensor
                 ) -> Tensor:
        pass


class MemoryEncoderProtocol(Protocol):
    def __call__(self,
                 batch: MemoryBatch
                 ) -> Tuple[Tensor, Tensor]:
        pass


class GraphEncoderProtocol(Protocol):
    def __call__(self,
                 x_dict: dict[str:Tensor],
                 edge_index_dict: dict[str:Tensor]
                 ) -> Tensor:
        pass


class OldLinkPredictorProtocol(Protocol):
    def __call__(self,
                 z_dict: dict[str:Tensor],
                 edge_label_indices: dict[tuple:Tensor],
                 edge_types
                 ) -> dict[int: Tensor]:
        pass

class LinkPredictorProtocol(Protocol):
    def __call__(self,
                 src: Tensor,
                 dst: Tensor,
                 edge_labels: Tensor
                 ) -> Tensor:
        pass

class HeteroGraph(Protocol):
    def __init__(self):
        HeteroData().__init__()

