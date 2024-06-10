from message_encoder import HeteroMessageEncoder
from random import randint, seed
import torch

seed(10)


def generate_fake_data(record_cnt, n_edge_types, max_dim=5):
    # Parameters for the encoder

    num_edge_types = n_edge_types
    edge_dims = []
    node_dims = []
    edges = []
    zero_dim_edge_f = 0
    zero_dim_node_f = 0
    for i in range(num_edge_types):
        len_n = len(node_dims)

        # add new node half the time or if no nodes yet
        add_new_node = randint(0, 1)
        add_new_node = True if len_n < 2 else add_new_node

        if add_new_node:
            if zero_dim_node_f == 0:
                node_dims.append(0)
                zero_dim_node_f += 1
            else:
                node_dims.append(randint(0, max_dim))


        if randint(0, 1):
            if len_n < 2:
                src_node = len_n
                dst_node = len_n
            else:
                src_node = len_n - 1
                dst_node = randint(0, len_n - 1)
        else:
            if len_n < 2:
                src_node = len_n
                dst_node = len_n
            else:
                src_node = randint(0, len_n - 1)
                dst_node = len_n - 1

        if zero_dim_edge_f == 0:
            edge_dims.append(0)
            zero_dim_edge_f += 1
        else:
            edge_dims.append(randint(0, max_dim))
        edges.append((src_node, i, dst_node))

    edge_records = [edges[randint(0, len(edges) - 1)] for _ in range(record_cnt)]

    # Generate fake data
    entity_types = torch.randint(0, 2, (record_cnt,))  # 0 NODE, 1 EDGE
    action_types = torch.randint(0, 3, (record_cnt,))  # 0 CREATE, 1 UPDATE, 2 DELETE

    src_node_types = [e[0] for e in edge_records]
    edge_types = [e[1] for e in edge_records]
    dst_node_types = [e[2] for e in edge_records]

    src_features = [torch.randn(node_dims[n]) for n in src_node_types]
    edge_features = [torch.randn(edge_dims[e]) for e in edge_types]
    dst_features = [torch.randn(node_dims[n]) for n in dst_node_types]

    return {'entity_types': entity_types,
            'action_types': action_types,
            'src_node_types': src_node_types,
            'src_features': src_features,
            'edge_types': edge_types,
            'edge_features': edge_features,
            'dst_node_types': dst_node_types,
            'dst_features': dst_features,
            'schema': {'node_dims': node_dims,
                       'edge_dims': edge_dims,
                       'edges': edges
                       }
            }


def test_msg_enc(record_cnt):
    records = generate_fake_data(record_cnt, 10)
    emb_dim = 12

    dropout = 0.1
    n_head = 1
    mlp_expansion_factor = 2
    bias = True

    schema = records['schema']
    del records['schema']

    # Instantiate the encoder
    encoder = HeteroMessageEncoder(
        emb_dim=emb_dim,
        node_dims=schema['node_dims'],
        edge_dims=schema['edge_dims'],
        edge_types=schema['edges'],
        dropout=dropout,
        n_head=n_head,
        mlp_expansion_factor=mlp_expansion_factor,
        bias=bias
    )

    print(encoder)

    # Process through the encoder
    messages = encoder(**records)

    return messages


if __name__ == '__main__':
    #print(generate_fake_data(10, 3, 5))
    msgs = test_msg_enc(100)
    print(msgs)
    print(msgs.shape)
