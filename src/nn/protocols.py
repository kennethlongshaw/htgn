from torch import Tensor, Optional
import torch
from typing import Protocol, Tuple, get_type_hints, List
from dataclasses import dataclass, fields
from torch_geometric.data import HeteroData
from itertools import compress


@dataclass(slots=True)
class MemoryBatch:
    time: Optional[Tensor]

    entity_types: Tensor
    action_types: Tensor

    src_ids: Tensor
    src_node_types: Tensor
    src_features: list[Tensor]

    edge_types: Tensor
    edge_features: [List[Tensor]]

    dst_ids: Tensor
    dst_node_types: Tensor
    dst_features: list[Tensor]

    src_memories: Optional[Tensor] = None
    dst_memories: Optional[Tensor] = None

    rel_time: Optional[Tensor] = None
    rel_time_enc: Optional[Tensor] = None

    def append(self, batch: 'MemoryBatch') -> 'MemoryBatch':
        return MemoryBatch(
            time=torch.cat([self.time, batch.time]),
            entity_types=torch.cat([self.entity_types, batch.entity_types]),
            action_types=torch.cat([self.action_types, batch.action_types]),
            src_ids=torch.cat([self.src_ids, batch.src_ids]),
            src_node_types=torch.cat([self.src_node_types, batch.src_node_types]),
            src_features=self.src_features + batch.src_features,
            dst_ids=torch.cat([self.dst_ids, batch.dst_ids]),
            dst_node_types=torch.cat([self.dst_node_types, batch.dst_node_types]),
            dst_features=self.dst_features + batch.edge_features,
            edge_types=torch.cat([self.edge_types, batch.edge_types]),
            edge_features=self.edge_features + batch.edge_features,
            src_memories=torch.cat(
                [self.src_memories, batch.src_memories]) if self.src_memories is not None else batch.src_memories,
            dst_memories=torch.cat(
                [self.dst_memories, batch.dst_memories]) if self.dst_memories is not None else batch.dst_memories,
            rel_time=torch.cat(
                [self.rel_time, batch.rel_time]) if self.rel_time is not None else batch.rel_time,
            rel_time_enc=torch.cat(
                [self.rel_time_enc, batch.rel_time_enc]) if self.rel_time_enc is not None else batch.rel_time_enc,
        )

    def __add__(self, other):
        return self.append(other)

    def filter_by_ids(self, filter_ids, filter_target='dst'):
        type_hints = get_type_hints(self.__class__)
        filter_masks = {
            'src': lambda filter_id: self.src_ids == filter_id,
            'dst': lambda filter_id: self.dst_ids == filter_id,
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
    def __init__(self,
                 message_enc: MessageEncoderProtocol,
                 aggregator: AggregatorProtocol,
                 time_enc: TimeEncoderProtocol,
                 memory_enc: MemoryProtocol
                 ):
        # Message computation module
        self.msg_enc = message_enc

        # Batch agg module
        self.aggregator = aggregator

        # time encoding
        self.time_enc = time_enc

        # sequential model
        self.memory_enc = memory_enc

    def __call__(self,
                 batch: MemoryBatch
                 ) -> Tuple[Tensor, Tensor]:
        pass


class GraphEncoderProtocol(Protocol):
    def __call__(self,
                 x_dict: dict[str:Tensor],
                 edge_index_dict: dict[str:Tensor]
                 ) -> dict[str:Tensor]:
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
                 edge_labels: Optional[Tensor] = None
                 ) -> Tensor:
        pass
