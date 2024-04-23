import random
import torch
import torch_geometric.data as data


class HeteroTemporalGraphGenerator:
    def __init__(self, node_types, edge_types, max_time, directed, feature_sizes, min_records):
        self.node_types = node_types
        self.edge_types = edge_types
        self.max_time = max_time
        self.directed = directed
        self.feature_sizes = feature_sizes
        self.min_records = min_records
        self.graph = data.HeteroData()
        self.node_ids = {node_type: [] for node_type in node_types}
        self.edge_ids = {edge_type: [] for edge_type in edge_types}
        self.messages = []
        self.record_id = 1
        self.time = 0

    def generate_initial_population(self):
        for node_type in self.node_types:
            num_nodes = random.randint(1, 10)
            for _ in range(num_nodes):
                self._create_node(node_type)

        for edge_type in self.edge_types:
            num_edges = random.randint(1, 5)
            for _ in range(num_edges):
                self._create_edge(edge_type)

    def generate_records(self):
        while len(self.messages) < self.min_records:
            self.time += 1
            action_type = random.choice(['CREATE', 'UPDATE', 'DELETE'])
            entity_type = random.choice(['NODE', 'EDGE'])

            if entity_type == 'NODE':
                node_type = random.choice(self.node_types)
                if action_type == 'CREATE':
                    self._create_node(node_type)
                elif action_type == 'UPDATE' and len(self.node_ids[node_type]) > 0:
                    self._update_node(node_type)
                elif action_type == 'DELETE' and len(self.node_ids[node_type]) > 0:
                    self._delete_node(node_type)
            else:  # EDGE
                edge_type = random.choice(list(self.edge_types))
                if action_type == 'CREATE' and len(self.node_ids[edge_type[0]]) > 0 and len(
                        self.node_ids[edge_type[2]]) > 0:
                    self._create_edge(edge_type)
                elif action_type == 'UPDATE' and len(self.edge_ids[edge_type]) > 0:
                    self._update_edge(edge_type)
                elif action_type == 'DELETE' and len(self.edge_ids[edge_type]) > 0:
                    self._delete_edge(edge_type)

    def _create_node(self, node_type):
        node_id = len(self.node_ids[node_type]) + 1
        self.node_ids[node_type].append(node_id)
        features = torch.randn(self.feature_sizes[node_type])
        self.graph[node_type].x = torch.cat((self.graph[node_type].x, features.unsqueeze(0)), dim=0) if hasattr(
            self.graph[node_type], 'x') else features.unsqueeze(0)
        self.messages.append(
            [self.record_id, 'NODE', 'CREATE', node_id, features.tolist(), node_type, node_id, features.tolist(),
             node_type, node_id, [0] * self.feature_sizes['edge'], f'{node_type}_SELF_LOOP', self.time])
        self.record_id += 1

    def _create_edge(self, edge_type):
        edge_id = len(self.edge_ids[edge_type]) + 1
        self.edge_ids[edge_type].append(edge_id)
        src_id = random.choice(self.node_ids[edge_type[0]])
        dst_id = random.choice(self.node_ids[edge_type[2]])
        edge_features = torch.randn(self.feature_sizes['edge'])
        self.graph[edge_type].edge_index = torch.cat(
            (self.graph[edge_type].edge_index, torch.tensor([[src_id - 1], [dst_id - 1]])), dim=1) if hasattr(
            self.graph[edge_type], 'edge_index') else torch.tensor([[src_id - 1], [dst_id - 1]])
        self.graph[edge_type].edge_attr = torch.cat((self.graph[edge_type].edge_attr, edge_features.unsqueeze(0)),
                                                    dim=0) if hasattr(self.graph[edge_type],
                                                                      'edge_attr') else edge_features.unsqueeze(0)
        self.messages.append(
            [self.record_id, 'EDGE', 'CREATE', src_id, self.graph[edge_type[0]].x[src_id - 1].tolist(), edge_type[0],
             dst_id, self.graph[edge_type[2]].x[dst_id - 1].tolist(), edge_type[2], edge_id, edge_features.tolist(),
             '_'.join(edge_type), self.time])
        self.record_id += 1
        if not self.directed:
            self._create_reverse_edge(edge_type, edge_id, dst_id, src_id, edge_features)

    def _create_reverse_edge(self, edge_type, edge_id, src_id, dst_id, edge_features):
        reverse_edge_type = (edge_type[2], f'REVERSE_{edge_type[1]}', edge_type[0])
        if reverse_edge_type not in self.edge_ids:
            self.edge_ids[reverse_edge_type] = []
        self.edge_ids[reverse_edge_type].append(edge_id)
        self.messages.append(
            [self.record_id, 'EDGE', 'CREATE', src_id, self.graph[edge_type[2]].x[src_id - 1].tolist(), edge_type[2],
             dst_id, self.graph[edge_type[0]].x[dst_id - 1].tolist(), edge_type[0], edge_id, edge_features.tolist(),
             '_'.join(reverse_edge_type), self.time])
        self.record_id += 1

    def _update_node(self, node_type):
        node_id = random.choice(self.node_ids[node_type])
        old_features = self.graph[node_type].x[node_id - 1].tolist()
        new_features = torch.randn(self.feature_sizes[node_type])
        self.graph[node_type].x[node_id - 1] = new_features
        self.messages.append(
            [self.record_id, 'NODE', 'UPDATE', node_id, new_features.tolist(), node_type, node_id, old_features,
             node_type, node_id, [0] * self.feature_sizes['edge'], f'{node_type}_SELF_LOOP', self.time])
        self.record_id += 1
        self._update_node_edges(node_type, node_id, new_features)

    def _update_node_edges(self, node_type, node_id, new_features):
        for edge_type in self.edge_types:
            if edge_type[0] == node_type:
                edge_indices = (self.graph[edge_type].edge_index[0] == node_id - 1).nonzero(as_tuple=True)[0]
                for edge_index in edge_indices:
                    dst_id = self.graph[edge_type].edge_index[1][edge_index].item() + 1
                    dst_type = edge_type[2]
                    edge_id = self.edge_ids[edge_type][edge_index]
                    self.messages.append(
                        [self.record_id, 'EDGE', 'UPDATE', node_id, new_features.tolist(), node_type, dst_id,
                         self.graph[dst_type].x[dst_id - 1].tolist(), dst_type, edge_id,
                         self.graph[edge_type].edge_attr[edge_index].tolist(), '_'.join(edge_type), self.time])
                    self.record_id += 1

    def _delete_node(self, node_type):
        node_id = random.choice(self.node_ids[node_type])
        features = self.graph[node_type].x[node_id - 1].tolist()
        self.messages.append(
            [self.record_id, 'NODE', 'DELETE', node_id, features, node_type, node_id, features, node_type, node_id,
             [0] * self.feature_sizes['edge'], f'{node_type}_SELF_LOOP', self.time])
        self.record_id += 1
        self.node_ids[node_type].remove(node_id)
        self.graph[node_type].x = torch.cat((self.graph[node_type].x[:node_id - 1], self.graph[node_type].x[node_id:]),
                                            dim=0)
        self._delete_node_edges(node_type, node_id)

    def _delete_node_edges(self, node_type, node_id):
        for edge_type in self.edge_types:
            if edge_type[0] == node_type or edge_type[2] == node_type:
                edge_indices = ((self.graph[edge_type].edge_index[0] == node_id - 1) | (
                        self.graph[edge_type].edge_index[1] == node_id - 1)).nonzero(as_tuple=True)[0]
                for edge_index in edge_indices:
                    src_id = self.graph[edge_type].edge_index[0][edge_index].item() + 1
                    dst_id = self.graph[edge_type].edge_index[1][edge_index].item() + 1
                    edge_id = self.edge_ids[edge_type][edge_index]
                    self.messages.append(
                        [self.record_id, 'EDGE', 'DELETE', src_id, self.graph[edge_type[0]].x[src_id - 1].tolist(),
                         edge_type[0], dst_id, self.graph[edge_type[2]].x[dst_id - 1].tolist(), edge_type[2], edge_id,
                         self.graph[edge_type].edge_attr[edge_index].tolist(), '_'.join(edge_type), self.time])
                    self.record_id += 1
                self.edge_ids[edge_type] = [edge_id for i, edge_id in enumerate(self.edge_ids[edge_type]) if
                                            i not in edge_indices]
                self.graph[edge_type].edge_index = torch.cat((self.graph[edge_type].edge_index[:, :edge_indices[0]],
                                                              self.graph[edge_type].edge_index[:,
                                                              edge_indices[-1] + 1:]), dim=1)
                self.graph[edge_type].edge_attr = torch.cat((self.graph[edge_type].edge_attr[:edge_indices[0]],
                                                             self.graph[edge_type].edge_attr[edge_indices[-1] + 1:]),
                                                            dim=0)

    def _update_edge(self, edge_type):
        edge_index = random.choice(range(self.graph[edge_type].edge_index.size(1)))
        src_id = self.graph[edge_type].edge_index[0][edge_index].item() + 1
        dst_id = self.graph[edge_type].edge_index[1][edge_index].item() + 1
        edge_id = self.edge_ids[edge_type][edge_index]
        old_features = self.graph[edge_type].edge_attr[edge_index].tolist()
        new_features = torch.randn(self.feature_sizes['edge'])
        self.graph[edge_type].edge_attr[edge_index] = new_features
        self.messages.append(
            [self.record_id, 'EDGE', 'UPDATE', src_id, self.graph[edge_type[0]].x[src_id - 1].tolist(), edge_type[0],
             dst_id, self.graph[edge_type[2]].x[dst_id - 1].tolist(), edge_type[2], edge_id, new_features.tolist(),
             '_'.join(edge_type), self.time])
        self.record_id += 1
        if not self.directed:
            self._update_reverse_edge(edge_type, edge_id, new_features)

    def _update_reverse_edge(self, edge_type, edge_id, new_features):
        reverse_edge_type = (edge_type[2], f'REVERSE_{edge_type[1]}', edge_type[0])
        reverse_edge_index = self.edge_ids[reverse_edge_type].index(edge_id)
        self.graph[reverse_edge_type].edge_attr[reverse_edge_index] = new_features
        dst_id = self.graph[reverse_edge_type].edge_index[0][reverse_edge_index].item() + 1
        src_id = self.graph[reverse_edge_type].edge_index[1][reverse_edge_index].item() + 1
        self.messages.append(
            [self.record_id, 'EDGE', 'UPDATE', dst_id, self.graph[edge_type[2]].x[dst_id - 1].tolist(), edge_type[2],
             src_id, self.graph[edge_type[0]].x[src_id - 1].tolist(), edge_type[0], edge_id, new_features.tolist(),
             '_'.join(reverse_edge_type), self.time])
        self.record_id += 1

    def _delete_edge(self, edge_type):
        edge_index = random.choice(range(self.graph[edge_type].edge_index.size(1)))
        src_id = self.graph[edge_type].edge_index[0][edge_index].item() + 1
        dst_id = self.graph[edge_type].edge_index[1][edge_index].item() + 1
        edge_id = self.edge_ids[edge_type][edge_index]
        features = self.graph[edge_type].edge_attr[edge_index].tolist()
        self.messages.append(
            [self.record_id, 'EDGE', 'DELETE', src_id, self.graph[edge_type[0]].x[src_id - 1].tolist(), edge_type[0],
             dst_id, self.graph[edge_type[2]].x[dst_id - 1].tolist(), edge_type[2], edge_id, features,
             '_'.join(edge_type), self.time])
        self.record_id += 1
        self.edge_ids[edge_type].remove(edge_id)
        self.graph[edge_type].edge_index = torch.cat(
            (self.graph[edge_type].edge_index[:, :edge_index], self.graph[edge_type].edge_index[:, edge_index + 1:]),
            dim=1)
        self.graph[edge_type].edge_attr = torch.cat(
            (self.graph[edge_type].edge_attr[:edge_index], self.graph[edge_type].edge_attr[edge_index + 1:]), dim=0)
        if not self.directed:
            self._delete_reverse_edge(edge_type, edge_id)

    def _delete_reverse_edge(self, edge_type, edge_id):
        reverse_edge_type = (edge_type[2], f'REVERSE_{edge_type[1]}', edge_type[0])
        reverse_edge_index = self.edge_ids[reverse_edge_type].index(edge_id)
        self.edge_ids[reverse_edge_type].remove(edge_id)
        self.graph[reverse_edge_type].edge_index = torch.cat((self.graph[reverse_edge_type].edge_index[:,
                                                              :reverse_edge_index],
                                                              self.graph[reverse_edge_type].edge_index[:,
                                                              reverse_edge_index + 1:]), dim=1)
        self.graph[reverse_edge_type].edge_attr = torch.cat((self.graph[reverse_edge_type].edge_attr[
                                                             :reverse_edge_index],
                                                             self.graph[reverse_edge_type].edge_attr[
                                                             reverse_edge_index + 1:]), dim=0)


def generate_valid_data(node_types, edge_types, max_time, directed, feature_sizes, min_records):
    generator = HeteroTemporalGraphGenerator(node_types, edge_types, max_time, directed, feature_sizes, min_records)
    generator.generate_initial_population()
    generator.generate_records()
    return generator.messages


def main():
    # Example usage
    node_types = ['User', 'Product']
    edge_types = [
        ('User', 'Purchases', 'Product'),
        ('User', 'Views', 'Product'),
        ('Product', 'ReviewedBy', 'User'),
        ('User', 'Follows', 'User')
    ]
    max_time = 100
    feature_sizes = {
        'User': 5,
        'Product': 3,
        'edge': 2
    }
    min_records = 100
    directed = True

    messages = generate_valid_data(node_types, edge_types, max_time, directed, feature_sizes, min_records)


if __name__ == '__main__':
    main()
