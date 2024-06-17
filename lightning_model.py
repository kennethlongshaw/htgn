from sklearn.metrics import average_precision_score, roc_auc_score
import pytorch_lightning as pl
import torch
from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import MessagePassing
from dataclasses import dataclass


class TemporalDataModule(pl.LightningDataModule):
    def __init__(self,
                 name: str,
                 path: str,
                 batch_size: int = 32,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 ):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.name = name

    def setup(self, stage: str) -> None:
        data = JODIEDataset(self.path, name=self.name)[0]
        self.train_data, self.val_data, self.test_data = data.train_val_test_split(val_ratio=self.val_ratio,
                                                                                   test_ratio=self.test_ratio)

    def train_dataloader(self):
        return TemporalDataLoader(
            self.train_data,
            batch_size=self.batch_size,
            neg_sampling_ratio=1.0,
        )

    def val_dataloader(self):
        return TemporalDataLoader(
            self.val_data,
            batch_size=self.batch_size,
            neg_sampling_ratio=1.0,
        )

    def test_dataloader(self):
        return TemporalDataLoader(
            self.test_data,
            batch_size=self.batch_size,
            neg_sampling_ratio=1.0,
        )


@dataclass
class HTGN_Config:
    num_nodes: int
    raw_msg_dim: int
    memory_dim: int
    time_dim: int
    msg_size: int
    bias: bool
    embedding_dim: int


class HTGN(pl.LightningModule):
    def __init__(self,
                 config: HTGN_Config,
                 message_encoder,
                 batch_agg,
                 memory,
                 gnn: MessagePassing,
                 link_pred,
                 feature_store,
                 graph_store,
                 criterion,
                 data
                 ):
        super().__init__()
        self.memory = memory(num_nodes=config.num_nodes,
                             raw_msg_dim=config.msg_size,
                             memory_dim=config.memory_dim,
                             time_dim=config.time_dim,
                             message_module=message_encoder(msg_dim=config.msg_size,
                                                            memory_dim=config.memory_dim,
                                                            time_dim=config.time_dim,
                                                            bias=config.bias
                                                            ),
                             aggregator_module=batch_agg
                             )
        self.gnn = gnn(in_channels=config.memory_dim,
                       out_channels=config.embedding_dim,
                       msg_dim=config.msg_size,
                       time_enc=self.memory.time_enc)
        self.link_pred = link_pred(in_channels=config.embedding_dim)
        self.feature_store = feature_store
        self.graph_store = graph_store
        self.assoc = torch.empty(config.num_nodes, dtype=torch.long)
        self.criterion = criterion
        self.data = data

    def model_step(self, batch):
        n_id, edge_index, e_id = self.history_loader(batch.n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

        z, last_update = self.memory(n_id)
        z = self.gnn(x=z,
                     last_update=last_update,
                     edge_index=edge_index,
                     t=self.data.t[e_id].to(self.device),
                     msg=self.data.msg[e_id].to(self.device)
                     )
        pos_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.dst]])
        neg_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.neg_dst]])

        return pos_out, neg_out

    def forward(self, batch):
        pos_out, neg_out = self.model_step(batch)

        loss = self.criterion(pos_out, torch.ones_like(pos_out))
        loss += self.criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        self.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        self.neighbor_loader.insert(batch.src, batch.dst)

        self.memory.detach()

    def validation_step(self, batch):
        pos_out, neg_out = self.model_step(batch)

        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        # Update memory and neighbor loader with ground-truth state.
        self.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        self.neighbor_loader.insert(batch.src, batch.dst)

    def clear_history(self):
        self.memory.reset_state()  # Start with a fresh memory.
        self.history_loader.reset_state()  # Start with an empty graph.

    def on_train_start(self) -> None:
        self.clear_history()

    def on_validation_start(self) -> None:
        self.clear_history()

    def on_test_start(self) -> None:
        self.clear_history()
