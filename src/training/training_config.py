from dataclasses import dataclass


@dataclass
class Training_Config:
    emb_dim: int
    dropout: float
    enc_heads: int
    enc_expansion: int
    bias: bool
    time_dim: int
    memory_dim: int
    num_layers: int
    agg: str = 'sum'
