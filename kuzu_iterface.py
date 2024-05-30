import kuzu
import os
import polars as pl


class KuzuInterface:
    def __init__(self,
                 db_path,
                 num_threads: int = os.cpu_count()
                 ):
        self._db = kuzu.Database(db_path)
        self._conn = kuzu.Connection(self._db, num_threads=num_threads)

    def __repr__(self):
        threads = self._conn.num_threads
        db_path = self._db.database_path
        return repr(f"Kuzu db @ {db_path} with {threads} threads")

    def get_query(self, query) -> pl.DataFrame:
        return self._conn.execute(query).get_as_pl()

    def get_table_info(self, table) -> pl.DataFrame:
        return self.get_query(f"CALL TABLE_INFO('{table}') RETURN *")

    def get_edge_info(self, edge) -> pl.DataFrame:
        return self.get_query(f"CALL show_connection('{edge}') RETURN *")

    def get_tables(self):
        return self.get_query("CALL show_tables() WHERE type = 'NODE' RETURN *")

    def get_edges(self):
        return self.get_query("CALL show_tables() WHERE type = 'REL' RETURN *")

    def get_schema(self) -> dict:
        nodes_tbls = self.get_tables()['name'].to_list()
        edges_tbls = self.get_edges()['name'].to_list()
        nodes = pl.concat([self.get_table_info(node).with_columns(node=pl.lit(node)) for node in nodes_tbls])
        edges = pl.concat([self.get_edge_info(edge).with_columns(edge=pl.lit(edge)) for edge in edges_tbls])
        return {'nodes': nodes, 'edges': edges}

    def create_schema(self, nodes: dict[str | dict[str:str]: str], edges: dict):

        if nodes is not None:
            for i in nodes.values():
                assert i['primary_key'] is not None

            for node, field_dict in nodes.items():
                stmt = f"""CREATE NODE TABLE {node.capitalize()} ({', '.join([f"{k.lower()} {v.upper()}" for k, v in field_dict['fields'].items()])}"""
                assert field_dict['primary_key'] in field_dict[
                    'fields'], f'{field_dict['primary_key']} not in field definitions'
                stmt = stmt + f", PRIMARY KEY ({field_dict['primary_key']}))"
                try:
                    print(f"creating {node.capitalize()}")
                    self._conn.execute(stmt)
                except RuntimeError as e:
                    if str(e) == f"Binder exception: {node.capitalize()} already exists in catalog.":
                        print(f"Node {node.capitalize()} already exists in catalog so creation was skipped.")
                    else:
                        # Handle other RuntimeError cases
                        print("An unexpected error occurred:", e)

        if edges is not None:
            for edge, node_pair in edges.items():
                assert node_pair[0] in nodes, f"Edge source {node_pair[0]} is not in list of nodes"
                assert node_pair[1] in nodes, f"Edge target {node_pair[1]} is not in list of nodes"
                stmt = f"CREATE REL TABLE {edge.upper()}(FROM {node_pair[0].capitalize()} TO {node_pair[1].capitalize()})"
                try:
                    print(f"creating {edge.upper()}")
                    self._conn.execute(stmt)
                except RuntimeError as e:
                    if str(e) == f"Binder exception: {edge.upper()} already exists in catalog.":
                        print(f"Edge {edge.upper()} already exists in catalog so creation was skipped.")
                    else:
                        # Handle other RuntimeError cases
                        print("An unexpected error occurred:", e)

    def drop_all_data(self):
        self._conn.execute("MATCH (n) DETACH DELETE n")

    def drop_schema(self):
        tables = self._conn.execute("CALL show_tables() RETURN *").get_as_df()
        for edge in self.edges:
            print(f"Dropping {edge.upper()}")
            self.conn.execute(f'DROP TABLE {edge.upper()}')
        for node in self.nodes:
            print(f"Dropping {node.capitalize()}")
            self.conn.execute(f'DROP TABLE {node.capitalize()}')

    def import_file(self, files: str | list[str], target):
        assert target in self.edges or target in self.nodes
        if isinstance(files, str) or isinstance(files, list):
            self.conn.execute(f'COPY {target} FROM "{files}"')
        else:
            assert isinstance(files, str) or isinstance(files, list), "Not a str or list or files"

    def insert_records(self, data):
        """BEGIN TRANSACTION;
        CREATE (a:User {name: 'Alice', age: 72});
        MATCH (a:User) RETURN *;
        COMMIT;"""

    def get_neighborhood(self,
                         node_ids: list,
                         k_hops: int = 1,
                         id_name: str = 'id'
                         ):
        return self._conn.execute(f"""MATCH (n)
                   WHERE n.{id_name} IN $nodes
                   MATCH (n)-[r*1..$k]-(m)
                   RETURN n, r, m""",
                                  parameters={'nodes': node_ids, 'k_hops': k_hops}).get_as_torch_geometric()
