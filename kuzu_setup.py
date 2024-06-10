from kuzu_iterface import KuzuInterface
import polars as pl
import kuzu
import torch_geometric
import torch
import time
def prep_schema(n_emb):
    nodes = {"user": {'fields': {'user_id': 'INT32', 'embeddings': f'DOUBLE[{n_emb}]'},
                      'primary_key': 'user_id'},
             'article': {'fields': {'article_id': 'INT32', 'embeddings': f'DOUBLE[{n_emb}]'},
                         'primary_key': 'article_id'},
             }
    edges = {'edits': ['user', 'article']}
    return {"nodes": nodes,
            "edges": edges
            }


def test_schema():
    nodes = {"User": {'fields':
                          {'registration_dttm': 'TIMESTAMP',
                           'id': 'INT32',
                           'first_name': 'STRING',
                           'last_name': 'STRING',
                           'email': 'STRING',
                           'gender': 'STRING',
                           'ip_address': 'STRING',
                           'cc': 'STRING',
                           'country': 'STRING',
                           'birthdate': 'STRING',
                           'salary': 'DOUBLE',
                           'title': 'STRING',
                           'comments': 'STRING',
                           },
                      'primary_key': 'id'},
             }
    return {"nodes": nodes,
            "edges": None
            }


def create_sample_db(conn: kuzu.connection.Connection):
    queries = [
        "CREATE NODE TABLE User(name STRING, age INT64, PRIMARY KEY (name))",
        'COPY User FROM "user.csv"',
        "CREATE NODE TABLE City(name STRING, population INT64, PRIMARY KEY (name))",
        'COPY City FROM "city.csv"',
        "CREATE REL TABLE Follows(FROM User TO User, since INT64)",
        'COPY Follows FROM "follows.csv"',
        "CREATE REL TABLE LivesIn(FROM User TO City)",
        'COPY LivesIn FROM "lives-in.csv"'
    ]
    for q in queries:
        print(q)
        conn.execute(q)


def load_bigger_sample_db(connection):
    connection.execute(
        'CREATE NODE TABLE Movie (movieId INT64, year INT64, title STRING, genres STRING, PRIMARY KEY (movieId))')
    connection.execute('CREATE NODE TABLE User (userId INT64, PRIMARY KEY (userId))')
    connection.execute('CREATE REL TABLE Rating (FROM User TO Movie, rating DOUBLE, timestamp INT64)')
    connection.execute('CREATE REL TABLE Tags (FROM User TO Movie, tag STRING, timestamp INT64)')

    connection.execute('COPY Movie FROM "./movies.csv" (HEADER=TRUE)')
    connection.execute('COPY User FROM "./users.csv" (HEADER=TRUE)')
    connection.execute('COPY Rating FROM "./ratings.csv" (HEADER=TRUE)')
    connection.execute('COPY Tags FROM "./tags.csv" (HEADER=TRUE)')


ku = KuzuInterface(db_path="./demo_db")

# ku.drop_all_data()
# ku.drop_schema()
# load_bigger_sample_db(ku._conn)
#print(ku.get_neighborhoods(node_type='Movie', node_ids=[1,2,3], k_hops=2, id_name='movieId'))
# query = """MATCH (n:Movie)
#                    WHERE n.movieId IN [1, 2, 3]
#                    MATCH (n)-[r*1..3]-(m)
#                    RETURN n, r, m"""
# print('starting')
# print(ku._conn.execute(query).get_as_torch_geometric()[0])
# create_sample_db(ku._conn)
# sch = test_schema()
# ku.create_schema(sch['nodes'], sch['edges'])
# df = pl.read_parquet('userdata1.parquet')
# ku.insert_pl_txn_nodes(df, 'User', load_as_one_txn=True)
# ku.drop_schema()

feature_store, graph_store = ku._db.get_torch_geometric_remote_backend()
from torch_geometric.loader import NeighborLoader

start_time = time.time()
num_tests = 10
for _ in range(num_tests):
    loader_kuzu = NeighborLoader(
        data=(feature_store, graph_store),
        num_neighbors={key.edge_type: [50] * 5 for key in graph_store.get_all_edge_attrs()},
        batch_size=1000,
        input_nodes=('Movie', list(range(1000))),

    )
    next(iter(loader_kuzu))
end_time = time.time()
total_time = end_time - start_time
average_time = total_time / num_tests
print(f"Total time to instantiate the loader {num_tests} times: {total_time:.2f} seconds")
print(f"Average time per loader instantiation: {average_time:.4f} seconds")
print(next(iter(loader_kuzu)))