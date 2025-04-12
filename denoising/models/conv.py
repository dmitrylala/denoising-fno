import torch
from neuralop.layers.spectral_convolution import _contract_tucker
from tltorch.factorized_tensors.core import FactorizedTensor
from torch import nn

__all__ = [
    'SpectralConv',
]


class SpectralConv(nn.Module):
    """
    Generic N-Dimensional Fourier Neural Operator.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels
    out_channels : int, optional
        Number of output channels
    n_modes : int tuple
        total number of modes to keep in Fourier Layer, along each dim
    rank : float or rank, optional
        Rank of the tensor factorization of the Fourier weights, by default 0.5
        Ignored if ``factorization is None``
    fft_norm : str, optional
        by default 'backward'

    """

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int | tuple[int],
        factorization: str,
        bias: bool = True,
        rank: float = 0.5,
        fft_norm: str = 'forward',
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # n_modes is the total number of modes kept along each dimension
        self.n_modes = n_modes
        self.order = len(self.n_modes)

        self.max_n_modes = self.n_modes
        self.fft_norm = fft_norm

        init_std = (2 / (in_channels + out_channels)) ** 0.5

        weight_shape = (in_channels, out_channels, *self.max_n_modes)

        # Create/init spectral weight tensor
        self.weight = FactorizedTensor.new(
            weight_shape,
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=None,
            dtype=torch.cfloat,
        )
        self.weight.normal_(0, init_std)

        self._contract = _contract_tucker

        self.bias = None
        if bias:
            self.bias = nn.Parameter(
                init_std * torch.randn(*((self.out_channels,) + (1,) * self.order)),
            )

    @property
    def n_modes(self) -> int | tuple[int]:
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes: int | tuple[int]) -> None:
        if isinstance(n_modes, int):  # Should happen for 1D FNO only  # noqa: SIM108
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        # the real FFT is skew-symmetric, so the last mode has a redundacy if our data is real in space  # noqa: E501
        # As a design choice we do the operation here to avoid users dealing with the +1
        # if we use the full FFT we cannot cut off informtion from the last mode
        n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generic forward pass for the Factorized Spectral Conv.

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)

        Returns
        -------
        tensorized_spectral_conv(x)

        """
        batchsize, _, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient in real spatial data

        # Compute Fourier coeffcients
        fft_dims = list(range(-self.order, 0))

        x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)

        if self.order > 1:
            x = torch.fft.fftshift(x, dim=fft_dims[:-1])

        out_fft = torch.zeros(
            [batchsize, self.out_channels, *fft_size],
            device=x.device,
            dtype=torch.cfloat,
        )

        # if current modes are less than max, start indexing modes closer to the center of the weight tensor  # noqa: E501
        starts = [
            (max_modes - min(size, n_mode))
            for (size, n_mode, max_modes) in zip(fft_size, self.n_modes, self.n_modes, strict=False)
        ]

        # weights have shape (in_channels, out_channels, modes_x, ...)
        slices_w = [slice(None), slice(None)]  # in_channels, out_channels

        # The last mode already has redundant half removed in real FFT
        slices_w += [
            slice(start // 2, -start // 2) if start else slice(start, None) for start in starts[:-1]
        ]
        slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]

        weight = self.weight[slices_w]

        # drop first two dims (in_channels, out_channels)
        weight_start_idx = 2
        starts = [
            (size - min(size, n_mode))
            for (size, n_mode) in zip(
                list(x.shape[2:]),
                list(weight.shape[weight_start_idx:]),
                strict=False,
            )
        ]
        slices_x = [slice(None), slice(None)]  # Batch_size, channels

        slices_x += [
            slice(start // 2, -start // 2) if start else slice(start, None) for start in starts[:-1]
        ]
        # The last mode already has redundant half removed
        slices_x += [slice(None, -starts[-1]) if starts[-1] else slice(None)]
        out_fft[slices_x] = self._contract(x[slices_x], weight)

        if self.order > 1:
            out_fft = torch.fft.fftshift(out_fft, dim=fft_dims[:-1])

        x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias

        return x
