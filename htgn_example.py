import set_determinism
import os.path as osp
from tqdm import tqdm
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear
from modified_tgn import AttentionTGN
from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    LastNeighborLoader,
)
from message_encoder import ExampleMessageTransformer, SumAggregator, MLPMessageEncoder, GraphAttention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'JODIE')
dataset = JODIEDataset(path, name='wikipedia')
data = dataset[0]

# For small datasets, we can put the whole dataset on GPU and thus avoid
# expensive memory transfer costs for mini-batches:
data = data.to(device)

add_degree_features = False

batch_size = 200

train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=0.15, test_ratio=0.15)

train_loader = TemporalDataLoader(
    train_data,
    batch_size=batch_size,
    neg_sampling_ratio=1.0,
)
val_loader = TemporalDataLoader(
    val_data,
    batch_size=batch_size,
    neg_sampling_ratio=1.0,
)
test_loader = TemporalDataLoader(
    test_data,
    batch_size=batch_size,
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
msg_size = data.msg.size(-1)  # plus 2 for src degree and dst degree
if add_degree_features:
    msg_size += 2

memory = TGNMemory(
    num_nodes=data.num_nodes,
    raw_msg_dim=msg_size,
    memory_dim=memory_dim,
    time_dim=time_dim,
    message_module=GraphAttention(msg_dim=msg_size,
                                  memory_dim=memory_dim,
                                  time_dim=time_dim,
                                  bias=False
                                  ),
    aggregator_module=SumAggregator()
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=msg_size - 2 * add_degree_features,
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters())
    | set(link_pred.parameters()), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


def train(add_degrees=False):
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    train_degrees = torch.zeros(data.num_nodes).to(device)

    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        if add_degrees:
            # added tracking for degree calc and state
            degrees = torch.bincount(torch.cat([batch.src, batch.dst]))
            padding = torch.zeros(train_degrees.shape[0] - degrees.shape[0], dtype=degrees.dtype).to(device)
            train_degrees += torch.cat([degrees, padding])
            src_degrees = train_degrees[batch.src].unsqueeze(1).detach()
            dst_degrees = train_degrees[batch.dst].unsqueeze(1).detach()

            batch_msg = torch.cat([batch.msg,
                                   src_degrees,
                                   dst_degrees
                                   ], dim=1)
        else:
            batch_msg = batch.msg

        # TODO: add tracking for common neighbors

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))
        pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(batch.src, batch.dst, batch.t, batch_msg)
        neighbor_loader.insert(batch.src, batch.dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events, train_degrees


@torch.no_grad()
def test(loader, add_degrees=False, degree_tensor: torch.Tensor = None):
    memory.eval()
    gnn.eval()
    link_pred.eval()

    test_degrees = degree_tensor
    aps, aucs = [], []
    for batch in loader:
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        if add_degrees:

            # added tracking for degree calc and state
            degrees = torch.bincount(torch.cat([batch.src, batch.dst]))
            padding = torch.zeros(test_degrees.shape[0] - degrees.shape[0], dtype=degrees.dtype).to(device)
            test_degrees += torch.cat([degrees, padding])
            src_degrees = test_degrees[batch.src].unsqueeze(1).detach()
            dst_degrees = test_degrees[batch.dst].unsqueeze(1).detach()

            batch_msg = torch.cat([batch.msg,
                                   src_degrees,
                                   dst_degrees
                                   ], dim=1)
        else:
            batch_msg = batch.msg

        # TODO: add tracking for common neighbors

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

        memory.update_state(batch.src, batch.dst, batch.t, batch_msg)
        neighbor_loader.insert(batch.src, batch.dst)
    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean()), test_degrees


def main():
    for epoch in tqdm(range(1, 51)):
        loss, train_degrees = train(add_degree_features)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        val_ap, val_auc, val_degrees = test(val_loader, add_degree_features, degree_tensor=train_degrees)
        test_ap, test_auc, _ = test(test_loader, add_degree_features, degree_tensor=val_degrees)
        print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')
        print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')


if __name__ == '__main__':
    main()
