import torch
import torch.nn as nn
from typing import Callable, Dict, Tuple, List
from torch import Tensor
import protocols as pr


class MemoryModule(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 message_module: pr.MessageEncoderProtocol,
                 aggregator_module: pr.AggregatorProtocol,
                 time_enc: pr.TimeEncoderProtocol,
                 memory_enc: pr.MemoryProtocol
                 ):
        super().__init__()
        self.num_nodes = num_nodes

        # Message prep module
        # Identity function in OG
        self.msg_module = message_module

        # Batch agg module
        # UseLast strategy in OG
        self.aggr_module = aggregator_module

        # time encoding
        self.time_enc = time_enc

        # sequential model
        self.memory_enc = memory_enc

    def forward(self,
                entity_types: Tensor,
                action_types: Tensor,

                src_node_types: list[int],
                src_features: list[Tensor],
                src_memories: Tensor,

                edge_types: list[int],
                edge_features: list[Tensor],

                dst_node_types: list[int],
                dst_features: list[Tensor],
                dst_memories: Tensor
                ):
        pass


class TimeEnc(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pr.Ti):

MemoryModule(time_enc=TimeEnc())
