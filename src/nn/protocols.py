from typing import List
from torch import Tensor
from typing import Protocol, Tuple
import memory_module as mm


class MessageEncoderProtocol(Protocol):
    def __call__(
            self,
            rel_t_enc: Tensor,
            entity_types: Tensor,
            action_types: Tensor,

            src_node_types: List[int],
            src_features: List[Tensor],
            src_memories: Tensor,

            edge_types: List[int],
            edge_features: List[Tensor],

            dst_node_types: List[int],
            dst_features: List[Tensor],
            dst_memories: Tensor
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
                 batch: mm.MemoryBatch,
                 dst_ids: Tensor
                 ) -> Tuple[Tensor, Tensor]:
        pass


class GraphEncoderProtocol(Protocol):
    def __call__(self, x_dict: dict[str:Tensor],
                 edge_index_dict: dict[str:Tensor]
                 ) -> Tensor:
        pass


class LinkPredictorProtocol(Protocol):
    def __call__(self,
                 z_dict: dict[str:Tensor],
                 edge_label_indices: dict[tuple:Tensor],
                 edge_types) -> dict[int: Tensor]:
        pass
