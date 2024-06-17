import torch.nn as nn
from torch import Tensor


class Transformer(nn.Module):
    """
        Implements a transformer block
    """

    def __init__(self,
                 emb_dim,
                 bias: bool = True,
                 dropout: float = 0.2,
                 num_heads: int = 1,
                 expansion_factor: int = 2):
        super().__init__()
        self.q_lin = nn.LazyLinear(emb_dim)
        self.k_lin = nn.LazyLinear(emb_dim)
        self.v_lin = nn.LazyLinear(emb_dim)
        self.attn = nn.MultiheadAttention(num_heads=num_heads,
                                          embed_dim=emb_dim,
                                          bias=bias,
                                          dropout=dropout
                                          )
        self.layer_norm = nn.LayerNorm(emb_dim, bias=bias)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * expansion_factor, bias=bias),
            nn.ReLU(),
            nn.Linear(emb_dim * expansion_factor, emb_dim, bias=bias),
            nn.Dropout(dropout)
        )

        self.attn_weight = None

    def forward(self, k: Tensor, q: Tensor, v: Tensor):
        # Determine whether to compute attention weights based on the mode
        need_weights = not self.training  # True if in eval mode, False if in training mode

        # Use src as both key and value in the attention mechanism
        attn_out, self.attn_weight = self.attn(query=self.q_lin(q), key=self.k_lin(k), value=self.v_lin(v),
                                               is_causal=False, attn_mask=None, need_weights=need_weights
                                               )
        x = q + attn_out.squeeze(0)  # Remove the batch dimension added for multi-head attention
        x = x + self.mlp(self.layer_norm(x))

        return x
