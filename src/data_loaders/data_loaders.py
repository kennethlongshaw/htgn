from torch.utils.data import IterableDataset
from src.nn.protocols import MemoryBatch
import polars as pl
from torch import Tensor
from typing import Optional, List


class PreppedDataset(IterableDataset):
    def __init__(self,
                 file_name: str,
                 sort_by: str = None,
                 descending: bool = None,
                 columns: list = None,
                 ):
        super().__init__()
        self.df = pl.read_parquet(file_name, columns=columns)
        if sort_by:
            self.df = self.df.sort(sort_by, descending=descending)

        self.schema = self.get_schema()

    def get_schema(self):
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
            raise NotImplemented()

        return {'node_dims': node_dims,
                'edge_types': edge_types,
                'edge_dims': edge_dims
                }

    def __iter__(self):
        return self.df.iter_rows(named=True)
