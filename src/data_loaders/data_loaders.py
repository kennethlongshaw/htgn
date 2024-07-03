from torch.utils.data import IterableDataset
import polars as pl
from typing import Optional
from dataclasses import dataclass


@dataclass
class GraphSchema:
    """
        Defines node and edge information about a graph
    """
    node_names: list[str]
    edge_names: list[tuple[str, str, str]]
    num_nodes: int
    node_dims: Optional[list[int]] = None
    edge_dims: Optional[list[int]] = None
    edge_types: Optional[list[tuple[int, int, int]]] = None


class PreppedDataset(IterableDataset):
    def __init__(self,
                 file_name: str,
                 sort_by: str = None,
                 descending: bool = None,
                 columns: list = None,
                 schema: GraphSchema = None
                 ):
        super().__init__()
        self.df = pl.read_parquet(file_name, columns=columns)
        if sort_by:
            self.df = self.df.sort(sort_by, descending=descending)

        self.validate_schema(schema)
        self.schema = schema

    def validate_schema(self, schema: GraphSchema):
        checked_schema_dims = self.get_schema_dims()
        if schema.edge_dims is not None:
            assert schema.edge_dims == checked_schema_dims['edge_dims']
        else:
            schema.edge_dims = checked_schema_dims['edge_dims']

        if schema.node_dims is not None:
            assert schema.node_dims == checked_schema_dims['node_dims']
        else:
            schema.node_dims = checked_schema_dims['node_dims']

        if schema.edge_types is not None:
            assert schema.edge_types == checked_schema_dims['edge_types']
        else:
            schema.edge_types = checked_schema_dims['edge_types']

    def get_schema_dims(self) -> dict[str, list]:
        node_dims = (pl
                     .concat([self.df.select(pl.col('src_types').alias('node'),
                                             pl.col('src_features').list.len().alias('dim')),
                              self.df.select(pl.col('dst_types').alias('node'),
                                             pl.col('dst_features').list.len().alias('dim')),
                              ])
                     .unique()
                     .sort('node')['dim'].to_list()
                     )
        edge_types = (self.df
                      .select(pl.col('src_types'), pl.col('edge_types'), pl.col('dst_types'))
                      .unique()
                      .sort('edge_types').rows())

        if 'edge_features' not in self.df.columns:
            edge_dims = [0 for _ in range(len(edge_types))]
        else:
            edge_dims = (self.df.select(pl.col('edge_types').alias('edge'),
                                        pl.col('edges_features').list.len().alias('dim')
                                        )
                         .unique()
                         .sort('edge')['dim'].to_list()
                         )

        num_nodes = (pl.concat([self.df.select('src_ids'), self.df.select('dst_ids')])
                     .unique()
                     .height
                     )

        return {'node_dims': node_dims,
                'edge_types': edge_types,
                'edge_dims': edge_dims,
                'num_nodes': num_nodes
                }

    def __iter__(self):
        return self.df.iter_rows(named=True)


def df_collate_fn(data: list) -> pl.DataFrame:
    return pl.DataFrame(data)
