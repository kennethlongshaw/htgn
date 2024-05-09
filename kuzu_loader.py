import kuzu
import os


class KuzuInterface:
    def __init__(self,
                 db_path,
                 num_threads: int = os.cpu_count()
                 ):
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db, num_threads=num_threads)

    def create_schema(self, nodes: dict[str | dict[str:str]: str], edges: dict):
        for i in nodes.items():
            assert i['primary_key'] is not None

        for node, field_dict in nodes.items():
            stmt = f"""CREATE NODE TABLE {node.capitalize()} ({', '.join([f"{k.lower()} {v.upper()}" for k, v in field_dict['fields'].items()])}"""
            assert field_dict['primary_key'] in field_dict[
                'fields'], f'{field_dict['primary_key']} not in field definitions'
            stmt = stmt + f", PRIMARY KEY ({field_dict['primary_key']}))"
            self.conn.execute(stmt)

        self.nodes = nodes

        for edge, node_pair in edges.items():
            assert node_pair[0] in nodes, f"Edge source {node_pair[0]} is not in list of nodes"
            assert node_pair[1] in nodes, f"Edge target {node_pair[1]} is not in list of nodes"
            stmt = f"CREATE REL TABLE {edge}(FROM {node_pair[0].capitalize()} TO {node_pair[1].capitalize()})"
            self.conn.execute(stmt)

        self.edges = edges

    def drop_data(self):
        self.conn.execute("MATCH (n) DETACH DELETE n")

    def drop_schema(self):
        for edge in self.edges:
            self.conn.execute(f'DROP TABLE {edge}')
        for node in self.nodes:
            self.conn.execute(f'DROP TABLE {node}')

    def get_neighborhood(self,
                         node_ids: list,
                         k_hops: int = 1,
                         id_name: str = 'id'
                         ):
        return self.conn.execute(f"""MATCH (n)
                   WHERE n.{id_name} IN $nodes
                   MATCH (n)-[r*1..$k]-(m)
                   RETURN n, r, m""",
                                 parameters={'nodes': node_ids, 'k_hops': k_hops}).get_as_torch_geometric()
