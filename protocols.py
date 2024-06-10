from typing import List
from torch import Tensor
from typing import Protocol


class MessageEncoderProtocol(Protocol):
    def __call__(
            self,
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

    ) -> None:
        pass


class AggregatorProtocol(Protocol):
    def __call__(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int) -> Tensor:
        pass


class TimeEncoderProtocol(Protocol):
    def __call__(self, t: Tensor):
        pass


class MemoryProtocol(Protocol):
    def __call__(self,
                 input_tensor: Tensor,
                 hidden_state: Tensor
                 ):
        pass
