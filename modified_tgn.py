from torch_geometric.nn.models import TGNMemory
from torch_geometric.nn.models.tgn import TimeEncoder
from message_encoder import AttentionMessageMemory
import copy
from typing import Callable, Dict, Tuple
import torch
from torch import Tensor
from torch.nn import GRUCell, Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import scatter
from torch_geometric.utils._scatter import scatter_argmax
from kuzu_iterface import KuzuInterface


class AttentionTGN(TGNMemory):
    """Overrides the memory module of TGN to use a cross attention-based memory system"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gru = AttentionMessageMemory(input_channels=kwargs['message_module'].out_channels,
                                          emb_dim=kwargs['memory_dim'])


# Message store format: id: (src, dst, t, msg)
TGNMessageStoreType = Dict[int, Tuple[Tensor, Tensor, Tensor, Tensor]]


class TGNMemory(torch.nn.Module):
    def __init__(self,
                 num_nodes: int,
                 memory_dim: int,
                 time_dim: int,
                 message_module: Callable,
                 aggregator_module: Callable,
                 ):
        super().__init__()
        self.num_nodes = num_nodes

        # Message prep module
        # Identity function in OG
        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)

        # Batch agg module
        # UseLast strategy in OG
        self.aggr_module = aggregator_module

        # time encoding
        self.time_enc = TimeEncoder(time_dim)

        # sequential model
        self.gru = GRUCell(message_module.out_channels, memory_dim)

    def forward(self,
                n_id: Tensor,
                nid_prev_memory: Tensor,
                src_msg_records: Tensor,
                dst_msg_records: Tensor,
                src_prev_memory: Tensor,
                dst_prev_memory: Tensor,
                last_update: Tensor,
                assoc: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp.
        """

        memory, last_update, assoc = self._get_updated_memory(n_id=n_id,
                                                              src_msg_records=src_msg_records,
                                                              dst_msg_records=dst_msg_records,
                                                              src_prev_memory=src_prev_memory,
                                                              dst_prev_memory=dst_prev_memory,
                                                              last_update=last_update,
                                                              assoc=assoc,
                                                              nid_prev_memory=nid_prev_memory
                                                              )

        return memory, last_update, assoc

    def _get_updated_memory(self,
                            n_id: Tensor,
                            nid_prev_memory: Tensor,
                            src_msg_records: Tensor,
                            dst_msg_records: Tensor,
                            src_prev_memory: Tensor,
                            dst_prev_memory: Tensor,
                            last_update: Tensor,
                            assoc: Tensor,
                            ) -> Tuple[Tensor, Tensor, Tensor]:
        assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Compute message takes in [src_prev_memory,
        #                          dst_prev_memory,
        #                          raw_msg,
        #                          rel_t_enc]
        # to form message

        # Compute messages (src -> dst).
        msg_s, t_s, src_s, dst_s = self._compute_msg(msg_records=src_msg_records,
                                                     msg_module=self.msg_s_module,
                                                     last_update=last_update,
                                                     src_prev_memory=src_prev_memory,
                                                     dst_prev_memory=dst_prev_memory
                                                     )

        # Compute messages (dst -> src).
        msg_d, t_d, src_d, dst_d = self._compute_msg(msg_records=dst_msg_records,
                                                     msg_module=self.msg_d_module,
                                                     last_update=last_update.data,
                                                     src_prev_memory=src_prev_memory,
                                                     dst_prev_memory=dst_prev_memory
                                                     )

        # Concat to prep
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)

        # Aggregate messages
        # Results in one agg message per node
        aggr = self.aggr_module(msg=msg, index=assoc[idx], t=t, dim_size=n_id.size(0))

        # Calc updated memory
        updated_memory = self.gru(input=aggr, hidden=nid_prev_memory)

        # Calc new last update for each node by agg max
        updated_last_update = scatter(src=t,
                                      index=idx,
                                      dim=0,
                                      dim_size=self.num_nodes,
                                      reduce='max')[n_id]

        return updated_memory, updated_last_update, assoc

    def _compute_msg(self,
                     msg_records: Tensor,
                     last_update: Tensor,
                     src_prev_memory: Tensor,
                     dst_prev_memory: Tensor,
                     msg_module: Callable):

        # break out vars from data and format/prep
        src, dst, t, raw_msg = list(zip(*msg_records))
        src = torch.cat(src, dim=0).to(self.device)
        dst = torch.cat(dst, dim=0).to(self.device)
        t = torch.cat(t, dim=0).to(self.device)

        # Filter out empty tensors to avoid `invalid configuration argument`.
        # TODO Investigate why this is needed.
        raw_msg = [m for i, m in enumerate(raw_msg) if m.numel() > 0 or i == 0]
        raw_msg = torch.cat(raw_msg, dim=0).to(self.device)

        # Calculates time since last update
        rel_t = t - last_update
        rel_t_enc = self.time_enc(rel_t.to(raw_msg.dtype))

        # gets memory of src and dst nodes to form message
        # passes message through message module
        msg = msg_module(src_prev_memory,
                         dst_prev_memory,
                         raw_msg,
                         rel_t_enc
                         )

        return msg, t, src, dst
