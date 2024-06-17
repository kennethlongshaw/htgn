import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor
from src.nn import protocols as pr
from dataclasses import dataclass, asdict


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


class MemoryModule(nn.Module):
    def __init__(self,
                 message_enc: pr.MessageEncoderProtocol,
                 aggregator: pr.AggregatorProtocol,
                 time_enc: pr.TimeEncoderProtocol,
                 memory_enc: pr.MemoryProtocol
                 ):
        super().__init__()
        # Message computation module
        self.msg_enc = message_enc

        # Batch agg module
        self.aggregator = aggregator

        # time encoding
        self.time_enc = time_enc

        # sequential model
        self.memory_enc = memory_enc

    def forward(self, batch: MemoryBatch, dst_ids: Tensor) -> Tuple[Tensor, Tensor]:
        # convert the relative time encoding
        rel_time = batch.rel_time
        del batch.rel_time
        batch.rel_time_enc = self.time_enc(rel_time)

        msg = self.msg_enc(**asdict(batch))

        with torch.no_grad():
            # map dst IDs to temporary index
            unique_ids, index = torch.unique(dst_ids, return_inverse=True)
            # get the first seen copy of each node memory
            mem_idx, _ = torch.unique(index, return_inverse=True)

        agg = self.aggregator(index=index, values=msg, num_nodes=unique_ids.size(0))

        prev_memories = batch.dst_memories[mem_idx]
        memories = self.memory_enc(hidden_state=prev_memories, input_tensor=agg)
        return unique_ids, memories
