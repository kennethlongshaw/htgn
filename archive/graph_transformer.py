import torch.nn
import torch_geometric as geo
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.conv import MessagePassing
from typing import Union, Tuple, Final
from torch import Tensor
from torch_geometric.typing import OptTensor, Adj
import torch.nn.functional as F

from typing import Optional, Callable, Any
from dataclasses import dataclass


class FlashTransformerConv(geo.nn.TransformerConv):
    """
        Implements the scaled dot product attention function in torch that can utilize flash attention v2 or the most
        efficient implementation that is available on the machine

        Should produce the same output on stock `torch_geometric.nn.TransformerConv` but faster on modern GPUs
    """

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            beta: bool = False,
            dropout: float = 0.,
            edge_dim: Optional[int] = None,
            bias: bool = True,
            root_weight: bool = True,
            **kwargs,
    ):
        super().__init__(in_channels, out_channels, heads=heads, concat=concat, beta=beta,
                         dropout=dropout, edge_dim=edge_dim, bias=bias, root_weight=root_weight, **kwargs)

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key_j = key_j + edge_attr
            value_j = value_j + edge_attr

        attn_output = F.scaled_dot_product_attention(
            query=query_i, key=key_j, value=value_j,
            attn_mask=None, dropout_p=self.dropout, is_causal=False,
        )

        # TODO: need to figure out weights, they can return with MHA but not SDPA
        #self._alpha = alpha.view(-1, self.heads)

        return attn_output


@dataclass
class GraphTransformerConfig:
    in_channels: Union[int, Tuple[int, int]]
    hidden_channels: int
    num_layers: int
    heads: int = 1
    concat: bool = True
    edge_dim: int = None
    beta: bool = False
    bias: bool = True
    root_weight: bool = True
    out_channels: Optional[int] = None
    dropout: Optional[float] = None
    act: Optional[str | Callable] = 'relu'
    act_first: Optional[bool] = None
    act_kwargs: Optional[dict] = None
    norm: Optional[str | Callable] = None
    norm_kwargs: Optional[dict[str, Any]] = None
    jk: Optional[str] = None


class GraphTransformerModel(BasicGNN):
    """
        Mirrors the way torch geometric creates models that only vary by their convolution choice
        Creates torch geometric model that uses our Flash Attention transformer convolution
    """

    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(self,
                  in_channels: Union[int, Tuple[int, int]],
                  out_channels: int,
                  **kwargs) -> MessagePassing:

        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)
        edge_dim = kwargs.pop('edge_dim', None)
        beta = kwargs.pop('beta', False)
        bias = kwargs.pop('bias', True)
        root_weight = kwargs.pop('root_weight', True)

        # Do not use concatenation in case the layer `TransformerConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'TransformerConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        return FlashTransformerConv(in_channels=in_channels,
                                    out_channels=out_channels,
                                    heads=heads,
                                    concat=concat,
                                    edge_dim=edge_dim,
                                    beta=beta,
                                    bias=bias,
                                    root_weight=root_weight,
                                    dropout=self.dropout.p,
                                    **kwargs)
