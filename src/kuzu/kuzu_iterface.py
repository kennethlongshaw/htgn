import kuzu
import os
import polars as pl
from loguru import logger


def format_timestamps(df: pl.DataFrame):
    for name, dtype in df.schema.items():
        if isinstance(dtype, pl.Datetime):
            if dtype.time_zone is None:
                # Format without timezone information if time_zone is None
                df = df.with_columns(
                    pl.col(name).dt.strftime('%Y-%m-%d %H:%M:%S.%f').alias(name)
                )
            else:
                # Format with timezone information if time_zone is present
                df = df.with_columns(
                    pl.col(name).dt.strftime('%Y-%m-%d %H:%M:%S.%f%z').alias(name)
                )
    return df


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

    def get_query_as_df(self, query) -> pl.DataFrame:
        return self._conn.execute(query).get_as_pl()

    def get_table_info(self, table) -> pl.DataFrame:
        return self.get_query_as_df(f"CALL TABLE_INFO('{table}') RETURN *")

    def get_edge_info(self, edge) -> pl.DataFrame:
        return self.get_query_as_df(f"CALL show_connection('{edge}') RETURN *")

    def get_tables(self):
        return self.get_query_as_df("CALL show_tables() WHERE type = 'NODE' RETURN *")

    def get_edges(self):
        return self.get_query_as_df("CALL show_tables() WHERE type = 'REL' RETURN *")

    def get_schema(self) -> dict:
        nodes_tbls = self.get_tables()['name'].to_list()
        edges_tbls = self.get_edges()['name'].to_list()
        nodes = pl.concat([self.get_table_info(node).with_columns(node=pl.lit(node)) for node in nodes_tbls])
        edges = pl.concat([self.get_edge_info(edge).with_columns(edge=pl.lit(edge)) for edge in edges_tbls])
        return {'nodes': nodes, 'edges': edges}

    def create_schema(self, nodes: dict[str | dict[str:str]: str], edges: dict):
        """
        Args:
            :param nodes: Node entries have a fields dict for name and dtype. Entries also have a dict of primary
        key and field name
            :param edges: Edge entries have a tuple of source table and target table
        """

        if nodes is not None:
            for i in nodes.values():
                assert i['primary_key'] is not None

            for node, field_dict in nodes.items():
                stmt = f"""CREATE NODE TABLE {node.capitalize()} ({', '.join([f"{k.lower()} {v.upper()}" for k, v in field_dict['fields'].items()])}"""
                assert field_dict['primary_key'] in field_dict[
                    'fields'], f'{field_dict['primary_key']} not in field definitions'
                stmt = stmt + f", PRIMARY KEY ({field_dict['primary_key']}))"
                try:
                    print(f"Creating {node.capitalize()}")
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
        nodes = self.get_tables()['name'].to_list()
        edges = self.get_edges()['name'].to_list()
        for edge in edges:
            print(f"Dropping {edge.upper()}")
            self._conn.execute(f'DROP TABLE {edge.upper()}')
        for node in nodes:
            print(f"Dropping {node.capitalize()}")
            self._conn.execute(f'DROP TABLE {node.capitalize()}')

    def import_parquet(self, files: str | list[str], target):
        assert target in self.get_edges()['name'].to_list() or target in self.get_tables()['name'].to_list()
        if isinstance(files, str) or isinstance(files, list):
            self._conn.execute(f'COPY {target} FROM "{files}"')
        else:
            assert isinstance(files, str) or isinstance(files, list), "Not a str or list or files"

    def import_pl_as_file(self, df: pl.DataFrame, target):
        temp_file_nm = 'temp_df_file_to_load'
        file_type = '.parquet'

        # avoid overwriting existing files
        cnt = 1
        while os.path.exists(temp_file_nm + file_type):
            temp_file_nm = temp_file_nm + str(cnt)
            cnt += 1

        df.write_parquet(temp_file_nm + file_type)
        self.import_file(temp_file_nm + file_type, target=target)
        os.remove(temp_file_nm + file_type)

    def insert_pl_txn_nodes(self,
                            df: pl.DataFrame,
                            target: str,
                            load_as_one_txn: bool = False):
        """Will load most simple data types including string, numeric, datatime and probably break on the rest which sucks because i really plan on using this for loading vectors"""
        time_cols = [k for k, v in df.schema.items() if isinstance(v, pl.Datetime)]
        if time_cols:
            # Format timestamp columns
            data = format_timestamps(df).rows(named=True)
        else:
            data = df.rows(named=True)

        if time_cols:
            for col in time_cols:
                for i, _ in enumerate(data):
                    data[i][col] = f'timestamp("{data[i][col]}")'

        stmts = []

        old_cnt = self.get_query_as_df(f"Match (n:{target}) return count(n)").row(0)[0]

        # Construct CREATE statements for each row
        errors = 0
        for i in data:
            properties = ", ".join([f'{k} : "{str(v)}"' if isinstance(v, str) and 'timestamp' not in v
                                    else f'{k} : {v}'
                                    for k, v in i.items() if v is not None
                                    and str(v).isascii()  # removing complex strings for now
                                    and str(v) != ""
                                    and '"' not in str(v)
                                    ])
            properties = "{" + properties + "}"
            stmt = f"CREATE (n:{target} {properties});"
            if not load_as_one_txn:
                try:
                    self._conn.execute(stmt)
                except Exception as e:
                    logger.error(f"{e} for statement ", stmt)
                    errors += 1

            else:
                stmts.append(stmt)
        if not load_as_one_txn:
            if errors == 0:
                logger.info(f"{errors} records failed to load")
            if errors > 0:
                logger.warning(f"{errors} records failed to load")
        else:
            bulk_stmt = "BEGIN TRANSACTION; " + " ".join(stmts) + " COMMIT;"
            self._conn.execute(bulk_stmt)

        new_cnt = self.get_query_as_df(f"Match (n:{target}) return count(n)").row(0)[0]
        logger.info(f"Count of nodes loaded to {target} is: {new_cnt - old_cnt}")

    def insert_pl_txn_edges(self,
                            df: pl.DataFrame,
                            edge: str,
                            load_as_one_txn: bool = False):
        """Assumes pl df where src and dst ids column names match primary keys in db"""
        e_info = self.get_edge_info('edits').rows(named=True)[0]
        data = df.rows(named=True)

        old_cnt = self.get_query(f"Match ()-[r:{edge}]-() return count(r)").row(0)[0]

        stmts = []

        # Construct CREATE statements for each row
        errors = 0
        for d in data:
            src_key = d[e_info['source table primary key']]
            dst_key = d[e_info['destination table primary key']]
            stmt = f"""
            MATCH (a:{e_info['source table name']}), (b:{e_info['destination table name']})
            WHERE a.{e_info['source table primary key']} == {src_key}
            and b.{e_info['destination table primary key']} == {dst_key}
            CREATE (a)-[:{edge}]->(b);
            """
            if not load_as_one_txn:
                try:
                    self._conn.execute(stmt)
                except Exception as e:
                    logger.error(f"{e} for statement ", stmt)
                    errors += 1

            else:
                stmts.append(stmt)

        if not load_as_one_txn:
            if errors == 0:
                logger.info(f"{errors} records failed to load")
            if errors > 0:
                logger.warning(f"{errors} records failed to load")
        else:
            bulk_stmt = "BEGIN TRANSACTION; " + " ".join(stmts) + " COMMIT;"
            self._conn.execute(bulk_stmt)

        new_cnt = self.get_query_as_df(f"Match ()-[r:{edge}]-() return count(r)").row(0)[0]
        logger.info(f"Count of edges loaded to {edge} is: {new_cnt - old_cnt}")

    def get_neighborhoods(self,
                          node_ids: list,
                          node_type: str,
                          k_hops: int = 1,
                          id_name: str = 'id'
                          ):
        # TODO: Add random sampling
        return self._conn.execute(f"""MATCH (n:{node_type.capitalize()})
                   WHERE n.{id_name} IN $nodes
                   MATCH (n)-[r*1..{k_hops}]-(m)
                   RETURN n, r, m""",
                                  parameters={'nodes': node_ids}).get_as_torch_geometric()
