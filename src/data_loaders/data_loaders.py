from torch.utils.data import IterableDataset, DataLoader
import polars as pl
from typing import Optional
from dataclasses import dataclass
import pytorch_lightning as lit
from src.utils.utils import df_to_batch
from src.nn.protocols import MemoryBatch

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


@dataclass
class DataModuleConfig:
    file_name: str
    time_column: str
    batch_size: int
    num_workers: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    descending: bool = None
    columns: list = None
    schema: GraphSchema = None


class PreppedDataset(IterableDataset):
    def __init__(self,
                 cfg: DataModuleConfig,
                 split: str = 'train'
                 ):
        super().__init__()
        self.df = pl.read_parquet(cfg.file_name, columns=cfg.columns)

        # Calculate split indices
        total_rows = len(self.df)
        train_end = int(total_rows * cfg.train_ratio)
        val_end = train_end + int(total_rows * cfg.val_ratio)

        self.df = self.df.sort(cfg.time_column, descending=cfg.descending)

        # Split the dataset
        if split == 'train':
            self.df = self.df.slice(0, train_end)
        elif split == 'val':
            self.df = self.df.slice(train_end, val_end - train_end)
        elif split == 'test':
            self.df = self.df.slice(val_end, total_rows - val_end)
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")

        self.validate_schema(cfg.schema)
        self.schema = cfg.schema

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


class GraphDataModule(lit.LightningDataModule):
    def __init__(self, cfg: DataModuleConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = PreppedDataset(cfg=self.cfg, split='train')
            self.val_dataset = PreppedDataset(cfg=self.cfg, split='val')
        if stage == 'test' or stage is None:
            self.test_dataset = PreppedDataset(cfg=self.cfg, split='test')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=df_collate_fn,
            shuffle=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=df_collate_fn,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=df_collate_fn,
            shuffle=False
        )


def df_collate_fn(data: list) -> MemoryBatch:
    return df_to_batch(pl.DataFrame(data))
