from pytorch_lightning.utilities.types import STEP_OUTPUT
from src.training.training_config import Training_Config
from src.nn import protocols as pr
import pytorch_lightning as lit
from torch import Tensor
from src.utils import utils
import torch.nn.functional as F
import torch
from typing import Tuple, Any, Optional
from src.data_stores.stores import MemoryStore, MessageStore, LastUpdateStore, EdgeStore


class HTGN(lit.LightningModule):
    def __init__(self,
                 mem_enc: pr.MemoryEncoderProtocol,
                 gnn_enc: pr.GraphEncoderProtocol,
                 link_pred: pr.LinkPredictorProtocol,
                 num_nodes: int,
                 train_cfg: Training_Config
                 ):
        super().__init__()
        self.mem_enc = mem_enc
        self.gnn_enc = gnn_enc
        self.link_pred = link_pred
        self.criterion = F.binary_cross_entropy
        self.num_nodes = num_nodes
        self.train_cfg = train_cfg

        self.graph_data = None
        self.neighbor_batch = None
        self.mem_store = None
        self.last_update_store = None
        self.edge_store = None
        self.msg_store = None

    def setup(self, stage: str) -> None:
        self.msg_store = MessageStore()
        self.mem_store = MemoryStore(num_nodes=self.num_nodes, memory_dim=self.train_cfg.memory_dim)
        self.last_update_store = LastUpdateStore(num_nodes=self.num_nodes)
        self.edge_store = EdgeStore()

    def on_train_epoch_start(self) -> None:
        """Cleans up any remaining data from previous epoch"""
        self.graph_data = None
        self.last_update_store.reset_state()
        self.msg_store.reset_state()
        self.mem_store.reset_state()
        self.edge_store = EdgeStore()
        self.neighbor_batch = None

    def on_train_batch_start(self, batch: pr.MemoryBatch, batch_idx: int) -> None:
        batch_nodes = torch.concat([batch.src_ids, batch.dst_ids, batch.neg_ids]).unique()

        # graph data
        self.graph_data = self.edge_store.get_edges(batch_nodes)
        self.graph_data.rel_t = self.last_update_store.calc_relative_time(self.graph_data.dst_ids,
                                                                          self.graph_data.times)

        # data for memory calc
        all_neighbors = torch.concat([self.graph_data.src_ids, self.graph_data.dst_ids]).unique()
        self.neighbor_batch = self.msg_store.get_from_msg_store(all_neighbors)
        self.neighbor_batch.rel_time = self.last_update_store.calc_relative_time(self.neighbor_batch.dst_ids,
                                                                                 self.neighbor_batch.time
                                                                                 )
        self.neighbor_batch.dst_memories = self.mem_store.get_memory(batch.dst_ids)
        self.neighbor_batch.src_memories = self.mem_store.get_memory(batch.src_ids)

    def on_train_batch_end(self,
                           outputs: STEP_OUTPUT,
                           batch: pr.MemoryBatch,
                           batch_idx: int) -> None:
        self.last_update_store.set_last_update(batch.dst_ids, batch.time)
        self.mem_store.set_memory(dst_ids=batch.dst_ids,
                                  memory=self.mem_enc(batch))
        self.msg_store.set_msg_store(batch=batch)

    def training_step(self, batch: pr.MemoryBatch) -> Tensor:
        self.graph_data.rel_t_enc = self.mem_enc.time_enc(self.graph_data.rel_t)
        neighbor_mem_ids, neighbor_mem = self.mem_enc(self.neighbor_batch)

        hgraph, mapping = utils.batch_to_graph(self.graph_data,
                                               memory_ids=neighbor_mem_ids,
                                               memories=neighbor_mem)

        z_dict = self.gnn_enc(x_dict=hgraph.x_dict,
                              edge_index_dict=hgraph.edge_index_dict)

        src_z = utils.zdict_lookup(mapping=mapping, z_dict=z_dict,
                                   ids=batch.src_ids, emb_dim=self.train_cfg.emb_dim)
        pos_dst_z = utils.zdict_lookup(mapping=mapping, z_dict=z_dict,
                                       ids=batch.dst_ids, emb_dim=self.train_cfg.emb_dim)
        neg_dst_z = utils.zdict_lookup(mapping=mapping, z_dict=z_dict,
                                       ids=batch.neg_ids, emb_dim=self.train_cfg.emb_dim)

        edge_labels = torch.cat([torch.ones_like(pos_dst_z),
                                 torch.zeros_like(neg_dst_z)
                                 ])

        pred = self.link_pred(src=torch.cat([src_z, src_z]),
                              dst=torch.cat([pos_dst_z, neg_dst_z]),
                              edge_labels=edge_labels
                              )

        loss = F.cross_entropy(pred, edge_labels)
        self.log('train_loss', loss)
        return loss

    def on_predict_batch_start(self, batch: pr.MemoryBatch, batch_idx: int, dataloader_idx: int = 0) -> None:
        batch_nodes = torch.concat([batch.src_ids, batch.dst_ids, batch.neg_ids]).unique()

        # graph data
        self.graph_data = self.edge_store.get_edges(batch_nodes)
        self.graph_data.rel_t = self.last_update_store.calc_relative_time(self.graph_data.dst_ids,
                                                                          self.graph_data.times)

    def predict_step(self, batch: pr.MemoryBatch) -> Tensor:
        src_ids, dst_ids = batch.src_ids, batch.dst_ids
        neighbor_mem_ids, neighbor_mem = self.mem_store.get_memories(src_ids, dst_ids)

        hgraph, mapping = utils.batch_to_graph(
            src_ids=src_ids,
            dst_ids=dst_ids,
            edge_types=self.graph_data.edge_types,
            edge_rel_t_enc=self.graph_data.rel_t_enc,
            edge_features=self.graph_data.edge_features,
            memory_ids=neighbor_mem_ids,
            memories=neighbor_mem
        )

        z_dict = self.gnn_enc(x_dict=hgraph.x_dict, edge_index_dict=hgraph.edge_index_dict)

        src_z = utils.zdict_lookup(mapping=mapping, z_dict=z_dict, ids=src_ids)
        dst_z = utils.zdict_lookup(mapping=mapping, z_dict=z_dict, ids=dst_ids)

        pred = self.link_pred(src=src_z, dst=dst_z)
        return pred

    def on_predict_batch_end(self,
                             outputs: Optional[Any],
                             batch: pr.MemoryBatch,
                             batch_idx: int,
                             dataloader_idx: int = 0) -> None:
        self.last_update_store.set_last_update(batch.dst_ids, batch.time)
        self.mem_store.set_memory(dst_ids=batch.dst_ids,
                                  memory=self.mem_enc(batch))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.tr_cfg.lr)
