from typing import Any

import torch
from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.embeddings import GridEmbeddingND
from neuralop.models.base_model import BaseModel
from torch import nn
from torch.nn.functional import gelu

from .fno_blocks import FNOBlocks

__all__ = [
    'FNO',
]


class FNO(BaseModel):
    """
    N-Dimensional Fourier Neural Operator. The FNO learns a mapping between
    spaces of functions discretized over regular grids using Fourier convolutions,
    as described in [1]_.

    The key component of an FNO is its SpectralConv layer (see
    ``neuralop.layers.spectral_convolution``), which is similar to a standard CNN
    conv layer but operates in the frequency domain.

    For a deeper dive into the FNO architecture, refer to :ref:`fno_intro`.

    Parameters
    ----------
    n_modes : Tuple[int]
        number of modes to keep in Fourier Layer, along each dimension
        The dimensionality of the FNO is inferred from ``len(n_modes)``
    in_channels : int
        Number of channels in input function
    out_channels : int
        Number of channels in output function
    hidden_channels : int
        width of the FNO (i.e. number of channels), by default 256
    n_layers : int, optional
        Number of Fourier Layers, by default 4

    Documentation for more advanced parameters is below.

    Other Parameters
    ----------------
    lifting_channel_ratio : int, optional
        ratio of lifting channels to hidden_channels, by default 2
        The number of liting channels in the lifting block of the FNO is
        lifting_channel_ratio * hidden_channels (e.g. default 512)
    projection_channel_ratio : int, optional
        ratio of projection channels to hidden_channels, by default 2
        The number of projection channels in the projection block of the FNO is
        projection_channel_ratio * hidden_channels (e.g. default 512)
    non_linearity : nn.Module, optional
        Non-Linear activation function module to use, by default F.gelu

    References
    ----------
    .. [1] :
    Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential
        Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.

    """  # noqa: D205

    def __init__(  # noqa: PLR0913
        self,
        n_modes: tuple[int],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        lifting_channel_ratio: int = 2,
        projection_channel_ratio: int = 2,
        non_linearity: nn.Module = gelu,
        rank: float = 0.42,
        **_: dict[str, Any],
    ) -> None:
        super().__init__()
        self.n_dim = len(n_modes)

        # n_modes is a special property - see the class' property for underlying mechanism
        # When updated, change should be reflected in fno blocks
        self._n_modes = n_modes

        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.n_layers = n_layers

        # init lifting and projection channels using ratios w.r.t hidden channels
        self.lifting_channels = lifting_channel_ratio * self.hidden_channels
        self.projection_channels = projection_channel_ratio * self.hidden_channels

        spatial_grid_boundaries = [[0.0, 1.0]] * self.n_dim
        self.positional_embedding = GridEmbeddingND(
            in_channels=self.in_channels,
            dim=self.n_dim,
            grid_boundaries=spatial_grid_boundaries,
        )

        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            non_linearity=non_linearity,
            n_layers=n_layers,
            rank=rank,
        )

        # if adding a positional embedding, add those channels to lifting
        lifting_in_channels = self.in_channels
        if self.positional_embedding is not None:
            lifting_in_channels += self.n_dim
        # if lifting_channels is passed, make lifting a Channel-Mixing MLP
        # with a hidden layer of size lifting_channels
        if self.lifting_channels:
            self.lifting = ChannelMLP(
                in_channels=lifting_in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
                non_linearity=non_linearity,
            )
        # otherwise, make it a linear layer
        else:
            self.lifting = ChannelMLP(
                in_channels=lifting_in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
                non_linearity=non_linearity,
            )

        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )

    def forward(self, x: torch.Tensor, **_: dict[str, Any]) -> torch.Tensor:
        """
        FNO's forward pass.

        1. Applies optional positional encoding

        2. Sends inputs through a lifting layer to a high-dimensional latent space

        3. Applies `n_layers` Fourier/FNO layers in sequence (SpectralConvolution + skip connections, nonlinearity)

        4. Projection of intermediate function representation to the output channels

        Parameters
        ----------
        x : tensor
            input tensor

        """  # noqa: E501
        # append spatial pos embedding if set
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lifting(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx)

        return self.projection(x)

    @property
    def n_modes(self) -> int | list[int]:
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes: int | list[int]) -> None:
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes
