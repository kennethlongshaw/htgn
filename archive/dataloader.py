


message = ['entity_type', 'action_type', 'src_id', 'src_features', 'dst_id', 'dst_features', 'edge_id', 'edge_features', 'edge_type', 'time']
# FEATURE ENRICHMENT
# need to track node degree over time as node features
# need to track common neighbors over time as edge features

import torch
from torch.utils.data import IterableDataset, DataLoader

class HeteroGraphMessageDataset(IterableDataset):
    def __init__(self, messages):
        self.messages = messages
        self.node_degrees = {}  # Dictionary to store node degrees by edge type and time
        self.common_neighbors = {}  # Dictionary to store common neighbors by edge type and time
        self.edge_types = set()  # Set to store unique edge types
        self.process_messages()  # Process messages to update node degrees and common neighbors

    def process_messages(self):
        for message in self.messages:
            entity_type, action_type, src_id, src_features, dst_id, dst_features, edge_id, edge_features, edge_type, time = message
            self.edge_types.add(edge_type)  # Add the edge type to the set of unique edge types

            if entity_type == 'NODE':
                if action_type == 'CREATE':
                    # Create a self-loop edge for the newly created node
                    if src_id not in self.node_degrees:
                        self.node_degrees[src_id] = {}
                    if 'SELF_LOOP' not in self.node_degrees[src_id]:
                        self.node_degrees[src_id]['SELF_LOOP'] = {}
                    self.node_degrees[src_id]['SELF_LOOP'][time] = 1

                elif action_type == 'DELETE':
                    # Remove the node from the node_degrees dictionary
                    if src_id in self.node_degrees:
                        del self.node_degrees[src_id]

            elif entity_type == 'EDGE':
                if action_type in ['CREATE', 'UPDATE']:
                    # Update node degrees by edge type
                    if src_id not in self.node_degrees:
                        self.node_degrees[src_id] = {}
                    if dst_id not in self.node_degrees:
                        self.node_degrees[dst_id] = {}
                    if edge_type not in self.node_degrees[src_id]:
                        self.node_degrees[src_id][edge_type] = {}
                    if edge_type not in self.node_degrees[dst_id]:
                        self.node_degrees[dst_id][edge_type] = {}
                    self.node_degrees[src_id][edge_type][time] = self.node_degrees[src_id][edge_type].get(time, 0) + 1
                    self.node_degrees[dst_id][edge_type][time] = self.node_degrees[dst_id][edge_type].get(time, 0) + 1

                    # Update common neighbors by edge type
                    edge_key = (src_id, dst_id, edge_type)
                    if edge_key not in self.common_neighbors:
                        self.common_neighbors[edge_key] = {}
                    for neighbor in self.node_degrees[src_id][edge_type]:
                        if neighbor in self.node_degrees[dst_id][edge_type]:
                            self.common_neighbors[edge_key][time] = self.common_neighbors[edge_key].get(time, 0) + 1

                elif action_type == 'DELETE':
                    # Update node degrees by edge type
                    if src_id in self.node_degrees and edge_type in self.node_degrees[src_id]:
                        if time in self.node_degrees[src_id][edge_type]:
                            self.node_degrees[src_id][edge_type][time] -= 1
                            if self.node_degrees[src_id][edge_type][time] == 0:
                                del self.node_degrees[src_id][edge_type][time]
                    if dst_id in self.node_degrees and edge_type in self.node_degrees[dst_id]:
                        if time in self.node_degrees[dst_id][edge_type]:
                            self.node_degrees[dst_id][edge_type][time] -= 1
                            if self.node_degrees[dst_id][edge_type][time] == 0:
                                del self.node_degrees[dst_id][edge_type][time]

                    # Update common neighbors by edge type
                    edge_key = (src_id, dst_id, edge_type)
                    if edge_key in self.common_neighbors:
                        if time in self.common_neighbors[edge_key]:
                            self.common_neighbors[edge_key][time] -= 1
                            if self.common_neighbors[edge_key][time] == 0:
                                del self.common_neighbors[edge_key][time]

    def __iter__(self):
        for message in self.messages:
            entity_type, action_type, src_id, src_features, dst_id, dst_features, edge_id, edge_features, edge_type, time = message

            # Get node degrees by edge type at the current time
            src_degree_features = [self.node_degrees.get(src_id, {}).get(et, {}).get(time, 0) for et in self.edge_types]
            dst_degree_features = [self.node_degrees.get(dst_id, {}).get(et, {}).get(time, 0) for et in self.edge_types]

            # Get common neighbors by edge type at the current time
            edge_key = (src_id, dst_id, edge_type)
            common_neighbors_features = [self.common_neighbors.get(edge_key, {}).get(time, 0)]

            # Create feature vectors
            src_features = torch.tensor(src_features + src_degree_features, dtype=torch.float)
            dst_features = torch.tensor(dst_features + dst_degree_features, dtype=torch.float)
            edge_features = torch.tensor(edge_features + common_neighbors_features, dtype=torch.float)

            yield entity_type, action_type, src_id, src_features, dst_id, dst_features, edge_id, edge_features, edge_type, time