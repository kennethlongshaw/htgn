from typing import List
from torch import Tensor
from typing import Protocol


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
