import torch.nn as nn
import torch
import torch.nn.functional as F

class ExampleMessageTransformer(nn.Module):
    """
        Implements a transformer block for message cross attention in order to aggregate messages to their destination
    """

    def __init__(self,
                 emb_dim: int,
                 memory_dim: int,
                 msg_dim: int,
                 time_dim: int,
                 bias: bool = True,
                 dropout: float = 0.2,
                 n_head: int = 1,
                 expansion_factor: int = 2,
                 ):
        super().__init__()
        self.src_linear = nn.Linear(memory_dim, emb_dim, bias=bias)
        self.msg_linear = nn.Linear(msg_dim, emb_dim, bias=bias)
        self.dst_linear = nn.Linear(memory_dim, emb_dim, bias=bias)

        self.src_norm = nn.LayerNorm(memory_dim, bias=bias)
        self.msg_norm = nn.LayerNorm(msg_dim, bias=bias)
        self.dst_norm = nn.LayerNorm(memory_dim, bias=bias)

        self.attn = nn.MultiheadAttention(num_heads=n_head,
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
        self.out_channels = emb_dim + time_dim

    def forward(self, src, dst, msg, t_enc):
        # norm and align
        src = self.src_linear(self.src_norm(src)) + self.msg_linear(self.msg_norm(msg))
        dst = self.dst_linear(self.dst_norm(dst))

        # Determine whether to compute attention weights based on the mode
        #need_weights = not self.training  # True if in eval mode, False if in training mode

        # Use src as both key and value in the attention mechanism
        attn_out, self.attn_weight = self.attn(query=dst.unsqueeze(0), key=src.unsqueeze(0), value=src.unsqueeze(0),
                                               is_causal=False, attn_mask=None
                                               )
        x = dst + attn_out.squeeze(0)  # Remove the batch dimension added for multi-head attention
        x = x + self.mlp(self.layer_norm(x))

        return torch.cat([x, t_enc], dim=-1)


class MLPMessageEncoder(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 memory_dim: int,
                 msg_dim: int,
                 time_dim: int,
                 bias: bool = True,
                 dropout: float = 0.2,
                 expansion_factor: int = 2,
                 ):
        super().__init__()
        self.src_norm = nn.LayerNorm(memory_dim, bias=bias)
        self.msg_norm = nn.LayerNorm(msg_dim, bias=bias)
        self.dst_norm = nn.LayerNorm(memory_dim, bias=bias)

        self.linear = nn.Linear(memory_dim * 2 + msg_dim, emb_dim, bias=bias)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * expansion_factor, bias=bias),
            nn.ReLU(),
            nn.Linear(emb_dim * expansion_factor, emb_dim, bias=bias),
            nn.Dropout(dropout)
        )

        self.out_channels = emb_dim + time_dim

    def forward(self, src, dst, msg, t_enc):
        # norm and align
        src = self.src_norm(src)
        msg = self.msg_norm(msg)
        dst = self.dst_norm(dst)
        x = torch.cat([src, dst, msg], dim=-1)
        x = self.linear(x)
        x = x + self.mlp(x)

        return torch.cat([x, t_enc], dim=-1)


class GraphAttention(nn.Module):
    def __init__(self,
                 memory_dim: int,
                 msg_dim: int,
                 time_dim: int,
                 bias: True
                 ):
        super(GraphAttention, self).__init__()

        self.src_norm = nn.LayerNorm(memory_dim, bias=bias)
        self.msg_norm = nn.LayerNorm(msg_dim, bias=bias)
        self.dst_norm = nn.LayerNorm(memory_dim, bias=bias)

        # Learnable parameters
        self.a_s = nn.Parameter(torch.randn(memory_dim, 1))
        self.a_t = nn.Parameter(torch.randn(memory_dim, 1))
        self.a_e = nn.Parameter(torch.randn(msg_dim, 1))
        self.leakyrelu = nn.LeakyReLU()

        self.out_channels = memory_dim + time_dim

    def forward(self, src, dst, msg, t_enc):

        dst = self.dst_norm(dst)

        # Calculate attention coefficients
        source_scores = self.leakyrelu(torch.matmul(self.src_norm(src), self.a_s))
        destination_scores = self.leakyrelu(torch.matmul(dst, self.a_t))
        msg_scores = self.leakyrelu(torch.matmul(self.msg_norm(msg), self.a_e))

        # Combine scores
        combined_scores = source_scores + destination_scores + msg_scores

        # Apply softmax to get attention weights
        attention_weights = F.softmax(combined_scores, dim=1)

        # Apply attention to destination features
        x = torch.mul(dst, attention_weights)

        return torch.cat([x, t_enc], dim=-1)


class AttentionMessageMemory(torch.nn.Module):
    def __init__(self, input_channels, emb_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_channels = input_channels
        self.emb_dim = emb_dim
        self.src_norm = nn.LayerNorm(input_channels, bias=True)
        self.src_linear = nn.Linear(input_channels, emb_dim, bias=True)

        self.hidden_norm = nn.LayerNorm(emb_dim, bias=True)

        self.attn = nn.MultiheadAttention(num_heads=1,
                                          embed_dim=emb_dim,
                                          bias=True,
                                          dropout=.1
                                          )
        self.layer_norm = nn.LayerNorm(emb_dim, bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2, bias=True),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim, bias=True),
        )

        self.attn_weight = None

    def forward(self, x, hidden_state):
        x = self.src_linear(self.src_norm(x))
        hidden_state = self.hidden_norm(hidden_state)
        # Use src as both key and value in the attention mechanism
        attn_out, self.attn_weight = self.attn(query=hidden_state.unsqueeze(0), key=x.unsqueeze(0),
                                               value=x.unsqueeze(0),
                                               is_causal=False, attn_mask=None
                                               )
        hidden_state = hidden_state + attn_out.squeeze(0)  # Remove the batch dimension added for multi-head attention
        return hidden_state + self.mlp(self.layer_norm(hidden_state))
