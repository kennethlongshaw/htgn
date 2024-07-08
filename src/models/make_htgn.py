from src.models.htgn import HTGN
from src.nn.encoders import (HeteroMessageEncoder_Config,
                             HeteroMessageEncoder,
                             TimeEncoder,
                             ScatterAggregator
                             )
from src.nn.memory_module import MemoryModule
from src.nn.decoders import DotProductLinkPredictor
from torch.nn import GRUCell
from torch_geometric.nn import to_hetero, GAT
from src.training.training_config import Training_Config
from src.data_loaders.data_loaders import GraphSchema


def make_HTGN(train_cfg: Training_Config,
              schema: GraphSchema):
    msg_enc_cfg = HeteroMessageEncoder_Config(
        emb_dim=train_cfg.emb_dim,
        node_dims=schema.node_dims,
        edge_dims=schema.edge_dims,
        edge_types=schema.edge_types,
        dropout=train_cfg.dropout,
        n_head=train_cfg.enc_heads,
        mlp_expansion_factor=train_cfg.enc_expansion,
        bias=train_cfg.bias,
        memory_dim=train_cfg.memory_dim,
        time_dim=train_cfg.time_dim
    )

    return HTGN(mem_enc=MemoryModule(message_enc=HeteroMessageEncoder(cfg=msg_enc_cfg),
                                     time_enc=TimeEncoder(train_cfg.time_dim),
                                     aggregator=ScatterAggregator(impl='geometric',
                                                                  reduce=train_cfg.agg
                                                                  ),
                                     memory_enc=GRUCell(input_size=train_cfg.emb_dim,
                                                        hidden_size=train_cfg.memory_dim
                                                        )
                                     ),
                gnn_enc=to_hetero(GAT(in_channels=-1,
                                      hidden_channels=train_cfg.emb_dim,
                                      num_layers=train_cfg.num_layers,
                                      add_self_loops=False,
                                      edge_dim=train_cfg.time_dim
                                      ),
                                  metadata=(schema.node_names, schema.edge_names)
                                  ),
                link_pred=DotProductLinkPredictor(),
                train_cfg=train_cfg,
                num_nodes=schema.num_nodes
                )
