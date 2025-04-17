import torch
from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.skip_connections import Flattened1dConv
from torch import nn
from torch.nn.functional import gelu

from .conv import (
    FourierSpectralConv,
    HartleySeparableSpectralConv,
    HartleySpectralConv,
)
from .soft_gating import SoftGating

__all__ = [
    'FNOBlocks',
]


class FNOBlocks(nn.Module):
    """
    FNOBlocks implements a sequence of Fourier layers, the operations of which
    are first described in [1]_. The exact implementation details of the Fourier
    layer architecture are discussed in [2]_.

    Parameters
    ----------
    in_channels : int
        input channels to Fourier layers
    out_channels : int
        output channels after Fourier layers
    n_modes : int, List[int]
        number of modes to keep along each dimension
        in frequency space. Can either be specified as
        an int (for all dimensions) or an iterable with one
        number per dimension
    n_layers : int, optional
        number of Fourier layers to apply in sequence, by default 1
    channel_mlp_dropout : float, optional
        dropout parameter for self.channel_mlp, by default 0
    channel_mlp_expansion : float, optional
        expansion parameter for self.channel_mlp, by default 0.5
    non_linearity : torch.nn.F module, optional
        nonlinear activation function to use between layers, by default F.gelu

    References
    ----------
    .. [1] Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential
           Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.
    .. [2] Kossaifi, J., Kovachki, N., Azizzadenesheli, K., Anandkumar, A. "Multi-Grid
           Tensorized Fourier Neural Operator for High-Resolution PDEs" (2024).
           TMLR 2024, https://openreview.net/pdf?id=AWiDlO63bH.

    """  # noqa: D205

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int | tuple[int],
        n_layers: int = 1,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        non_linearity: nn.Module = gelu,
        rank: float = 0.42,
        factorization: str = 'tucker',
        spectral: str = 'fourier',
    ) -> None:
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)

        self.n_layers = n_layers
        self.non_linearity = non_linearity

        conv_kwargs = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'n_modes': self.n_modes,
            'rank': rank,
            'factorization': factorization,
        }
        conv_module = None
        if spectral == 'fourier':
            conv_module = FourierSpectralConv
        elif spectral == 'hartley-separable':
            conv_module = HartleySeparableSpectralConv
        elif spectral == 'hartley':
            conv_module = HartleySpectralConv
        else:
            msg = f'Unknown spectral module: {spectral}'
            raise ValueError(msg)

        self.convs = nn.ModuleList(
            [conv_module(**conv_kwargs) for _ in range(n_layers)],
        )

        self.fno_skips = nn.ModuleList(
            [
                Flattened1dConv(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                )
                for _ in range(n_layers)
            ],
        )

        self.channel_mlp = nn.ModuleList(
            [
                ChannelMLP(
                    in_channels=out_channels,
                    hidden_channels=round(out_channels * channel_mlp_expansion),
                    dropout=channel_mlp_dropout,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ],
        )

        self.channel_mlp_skips = nn.ModuleList(
            [
                SoftGating(
                    in_channels,
                    out_channels,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ],
        )

    def forward(self, x: torch.Tensor, index: int = 0) -> torch.Tensor:
        x_skip_fno = self.fno_skips[index](x)
        x_skip_channel_mlp = self.channel_mlp_skips[index](x)
        x_fno = self.convs[index](x)

        x = x_fno + x_skip_fno

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        x = self.channel_mlp[index](x) + x_skip_channel_mlp

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        return x

    @property
    def n_modes(self) -> int | tuple[int]:
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes: int | tuple[int]) -> None:
        for i in range(self.n_layers):
            self.convs[i].n_modes = n_modes
        self._n_modes = n_modes
