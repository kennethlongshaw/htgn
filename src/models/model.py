from ..nn import memory_module as mm, encoders as enc
import pytorch_lightning as lit
import torch.nn as nn
from torch import Tensor

class HTGN(lit.LightningModule):
    def __init__(self,
                 mem_cfg: enc.HeteroMessageEncoder_Config,
                 gnn_cfg: dict
                 ):
        super().__init__()
        self.mem_enc = mm.MemoryModule(
            time_enc=enc.TimeEncoder(mem_cfg.time_dim),
            aggregator=enc.ScatterAggregator(impl='geometric', reduce='sum'),
            message_enc=enc.HeteroMessageEncoder(mem_cfg),
            memory_enc=nn.GRUCell(input_size=mem_cfg.emb_dim,
                                  hidden_size=mem_cfg.memory_dim
                                  )
        )

    def forward(self, batch: mm.MemoryBatch) -> Tensor:




