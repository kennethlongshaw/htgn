from ..nn import memory_module as mm, protocols as pr
import pytorch_lightning as lit
from torch import Tensor
from ..utils import utils
import torch.nn.functional as F


class HTGN(lit.LightningModule):
    def __init__(self,
                 mem_enc: pr.MemoryEncoderProtocol,
                 gnn_enc: pr.GraphEncoderProtocol,
                 link_pred: pr.LinkPredictorProtocol,
                 ):
        super().__init__()
        self.mem_enc = mem_enc
        self.gnn_enc = gnn_enc
        self.link_pred = link_pred
        self.criterion = F.binary_cross_entropy

    def forward(self,
                batch: mm.MemoryBatch,
                dst_ids: Tensor,
                src_ids: Tensor,
                neighbor_batch: mm.MemoryBatch,
                neighbor_dst_ids: Tensor,
                neighbor_src_ids: Tensor
                ) -> Tensor:
        neighbor_ids, neighbor_mem = self.mem_enc(neighbor_batch, neighbor_dst_ids)
        hgraph = utils.batch_to_graph(src_ids=neighbor_src_ids,
                                      dst_ids=neighbor_dst_ids,
                                      memory_ids=neighbor_ids,
                                      memories=neighbor_mem,
                                      batch=batch)
        z_dict = self.gnn_enc(x_dict=hgraph.x_dict, edge_index_dict=hgraph.edge_index_dict)

        pred = self.link_pred(z_dict=z_dict,
                              edge_label_indices=hgraph.edge_label_index,
                              edge_types=hgraph.metadata()[1])

        return pred

    def training_step(self,
                      batch: mm.MemoryBatch,
                      dst_ids: Tensor,
                      src_ids: Tensor,
                      neighbor_batch: mm.MemoryBatch,
                      neighbor_dst_ids: Tensor,
                      neighbor_src_ids: Tensor):
        pred = self(batch=batch,
                    dst_ids=dst_ids,
                    src_ids=src_ids,
                    neighbor_batch=neighbor_batch,
                    neighbor_dst_ids=neighbor_dst_ids,
                    neighbor_src_ids=neighbor_src_ids
                    )

        loss = self.criterion(pred, batch.edge_labels)
        batch_mem = self.mem_enc(batch, dst_ids)

        return loss, batch_mem
