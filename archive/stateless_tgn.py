import set_determinism
import os.path as osp

from tqdm import tqdm
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear
from torch import Tensor

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TransformerConv
from modified_tgn import TGNMessageStoreType, TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_path = '../data/'
path = osp.join(base_path, 'data', 'JODIE')
dataset = JODIEDataset(path, name='wikipedia')
data = dataset[0]

# For small datasets, we can put the whole dataset on GPU and thus avoid
# expensive memory transfer costs for mini-batches:
data = data.to(device)

train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=0.15, test_ratio=0.15)

train_loader = TemporalDataLoader(
    train_data,
    batch_size=200,
    neg_sampling_ratio=1.0,
)
val_loader = TemporalDataLoader(
    val_data,
    batch_size=200,
    neg_sampling_ratio=1.0,
)
test_loader = TemporalDataLoader(
    test_data,
    batch_size=200,
    neg_sampling_ratio=1.0,
)
neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


memory_dim = time_dim = embedding_dim = 100

memory = TGNMemory(
    num_nodes=data.num_nodes,
    memory_dim=memory_dim,
    time_dim=time_dim,
    message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters())
    | set(link_pred.parameters()), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss()


def update_msg_store(src: Tensor, dst: Tensor, t: Tensor,
                     raw_msg: Tensor, msg_store: TGNMessageStoreType):
    n_id, perm = src.sort()
    n_id, count = n_id.unique_consecutive(return_counts=True)
    for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
        msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])
    return msg_store


def train():
    msg_s_store = {}
    msg_d_store = {}

    memory.train()
    gnn.train()
    link_pred.train()

    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)

        neighbor_ids, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[neighbor_ids] = torch.arange(neighbor_ids.size(0), device=device)

        neighbor_src_msg_records = [msg_s_store.get(i) for i in neighbor_ids.tolist()]
        neighbor_dst_msg_records = [msg_d_store.get(i) for i in neighbor_ids.tolist()]

        neighbor_prev_memory = mem_store[neighbor_ids]

        # Get updated memory of all nodes involved in the computation.
        z, last_update, assoc = memory(n_id=neighbor_ids,
                                       src_msg_records=neighbor_src_msg_records,
                                       dst_msg_records=neighbor_dst_msg_records,
                                       last_update=last_update_store,
                                       assoc=assoc,
                                       nid_prev_memory=neighbor_prev_memory
                                       )

        # operates on graph of memories as nodes
        # encoded time since update and original message as edge features
        z = gnn(x=z,
                last_update=last_update,
                edge_index=edge_index,
                t=data.t[e_id].to(device),
                msg=data.msg[e_id].to(device)
                )

        pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        src_msg_records = [msg_s_store[i] for i in batch.n_id.tolist()]
        dst_msg_records = [msg_d_store[i] for i in batch.n_id.tolist()]

        batch_prev_memory = mem_store[batch.n_id]

        # update memory store
        new_mem, new_last_update, assoc = memory(n_id=batch.n_id,
                                                 src_msg_records=src_msg_records,
                                                 dst_msg_records=dst_msg_records,
                                                 last_update=last_update_store,
                                                 assoc=assoc,
                                                 nid_prev_memory=batch_prev_memory
                                                 )
        mem_store[batch.n_id] = new_mem
        last_update_store[batch.n_id] = new_last_update

        # update message store
        msg_s_store = update_msg_store(batch.src, batch.dst, batch.t, batch.raw_msg, msg_s_store)
        msg_d_store = update_msg_store(batch.dst, batch.src, batch.t, batch.raw_msg, msg_d_store)

        # update neighbor loader with edges
        neighbor_loader.insert(batch.src, batch.dst)

        loss.backward()
        optimizer.step()

        mem_store.detach()
        last_update_store.detach()
        assoc.detach()

        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


@torch.no_grad()
def test(loader):
    memory.eval()
    gnn.eval()
    link_pred.eval()

    aps, aucs = [], []

    for batch in loader:
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))
        pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())


for epoch in tqdm(range(1, 51)):
    # Helper vector to map global node indices to local ones.

    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
    mem_store = torch.empty(data.num_nodes, memory_dim).to(device)
    last_update_store = torch.empty(data.num_nodes, dtype=torch.long).to(device)

    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    val_ap, val_auc = test(val_loader)
    test_ap, test_auc = test(test_loader)
    print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')
    print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')
