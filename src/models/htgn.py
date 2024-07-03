from src.nn import protocols as pr
import pytorch_lightning as lit
from torch import Tensor
from src.utils import utils
import torch.nn.functional as F
import torch
from typing import Tuple
from src.data_stores.stores import MemoryStore, MessageStore, LastUpdateStore, EdgeStore


class HTGN(lit.LightningModule):
    def __init__(self,
                 mem_enc: pr.MemoryEncoderProtocol,
                 gnn_enc: pr.GraphEncoderProtocol,
                 link_pred: pr.LinkPredictorProtocol,
                 num_nodes: int,
                 memory_dim: int
                 ):
        super().__init__()
        self.mem_enc = mem_enc
        self.gnn_enc = gnn_enc
        self.link_pred = link_pred
        self.criterion = F.binary_cross_entropy
        self.msg_store = MessageStore()
        self.mem_store = MemoryStore(num_nodes=num_nodes, memory_dim=memory_dim)
        self.last_update_store = LastUpdateStore(num_nodes=num_nodes)
        self.edge_store = EdgeStore()

    def forward(self,
                batch: pr.MemoryBatch
                ) -> Tuple[Tensor, Tensor]:
        all_nodes = torch.concat([batch.src_ids, batch.dst_ids, batch.neg_ids]).unique()
        # todo: make edge store
        src_ids, dst_ids, edge_types, edge_features = self.edge_store.get_edges(all_nodes)

        all_neighbors = torch.concat([src_ids, dst_ids]).unique()

        neighbor_batch = self.msg_store.get_from_msg_store(all_neighbors)

        neighbor_batch.dst_memories = self.mem_store.get_memory(batch.dst_ids)
        neighbor_batch.src_memories = self.mem_store.get_memory(batch.src_ids)

        last_update = self.last_update_store.get_last_update(neighbor_batch.dst_ids)
        neighbor_batch.rel_time = self.last_update_store.calc_relative_time(last_update)
        neighbor_mem_ids, neighbor_mem = self.mem_enc(neighbor_batch)

        hgraph = utils.batch_to_graph(src_ids=src_ids,
                                      dst_ids=dst_ids,
                                      edge_types=edge_types,
                                      edge_features=edge_features,
                                      memory_ids=neighbor_mem_ids,
                                      memories=neighbor_mem)

        z_dict = self.gnn_enc(x_dict=hgraph.x_dict,
                              edge_index_dict=hgraph.edge_index_dict)

        src_z = utils.zdict_lookup(hgraph, z_dict, ids=batch.src_ids)
        pos_dst_z = utils.zdict_lookup(hgraph, z_dict, ids=batch.dst_ids)
        neg_dst_z = utils.zdict_lookup(hgraph, z_dict, ids=batch.neg_ids)

        edge_labels = torch.cat([torch.ones_like(pos_dst_z),
                                 torch.zeros_like(neg_dst_z)
                                 ])

        pred = self.link_pred(src=torch.cat([src_z, src_z]),
                              dst=torch.cat([pos_dst_z, neg_dst_z]),
                              edge_labels=edge_labels
                              )

        self.last_update_store.set_last_update(batch.dst_ids, batch.time)

        return pred, edge_labels

    def training_step(self,
                      batch: pr.MemoryBatch,
                      neighbor_batch: pr.MemoryBatch
                      ):
        pred, edge_label = self(batch=batch,
                                neighbor_batch=neighbor_batch
                                )

        loss = self.criterion(pred, edge_label)
        batch_mem = self.mem_enc(batch)

        return loss, batch_mem
