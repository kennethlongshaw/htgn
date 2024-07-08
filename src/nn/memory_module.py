import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor
from src.nn import protocols as pr


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

    def forward(self, batch: pr.MemoryBatch) -> Tuple[Tensor, Tensor]:
        # convert the relative time encoding
        batch.rel_time_enc = self.time_enc(batch.rel_time)
        msg = self.msg_enc(batch)

        with torch.no_grad():
            # map dst IDs to temporary index
            unique_ids, index = torch.unique(batch.dst_ids, return_inverse=True)
            # get the first seen copy of each node memory
            mem_idx, _ = torch.unique(index, return_inverse=True)

        agg = self.aggregator(index=index, values=msg, num_nodes=unique_ids.size(0))

        prev_memories = batch.dst_memories[mem_idx]
        memories = self.memory_enc(agg, prev_memories)
        return unique_ids, memories
