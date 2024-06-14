import torch_geometric.nn

from ..nn import memory_module as mm, encoders as enc, protocols as pr
import pytorch_lightning as lit
import torch.nn as nn
from torch import Tensor
from ..utils import utils

class HTGN(lit.LightningModule):
    def __init__(self,
                 mem_enc: pr.MemoryEncoderProtocol,
                 gnn_enc: pr.GraphEncoderProtocol,
                 link_pred: pr.LinkPredictorProtocol
                 ):
        super().__init__()
        self.mem_enc = mem_enc
        self.gnn_enc = gnn_enc
        self.link_pred = lambda x: x
        self.criterion = lambda x: x

    def forward(self,
                batch: mm.MemoryBatch,
                dst_ids: Tensor,
                neighbor_batch: mm.MemoryBatch,
                neighbor_dst_ids: Tensor,
                neighbor_edge_index: dict[Tensor],
                neighbor_edge_attr: dict[Tensor],
                ) -> Tensor:
        neighbor_mem = self.mem_enc(neighbor_batch, neighbor_dst_ids)
        hgraph = utils.make_graph(neighbor_mem, neighbor_edge_attr, neighbor_edge_index)
        node_emb = gnn(hgraph)
        loss = self.criterion(node_emb, hgraph.edge_labels)
        batch_mem = self.mem_enc(batch, dst_ids)

        return loss, batch_mem
