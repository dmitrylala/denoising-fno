import tensorly as tl
import torch
from neuralop.layers.spectral_convolution import _contract_tucker
from tensorly.plugins import use_opt_einsum
from tltorch.factorized_tensors.core import FactorizedTensor
from torch import nn

tl.set_backend('pytorch')
use_opt_einsum('optimal')

__all__ = [
    'SpectralConv2D',
]


class SpectralConv2D(nn.Module):
    """
    2D Fourier Neural Operator.

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

        self.fft_norm = fft_norm

        init_std = (2 / (in_channels + out_channels)) ** 0.5
        weight_shape = (in_channels, out_channels, *self.n_modes)

        # Create/init spectral weight tensor
        self.weight = FactorizedTensor.new(
            weight_shape,
            rank=rank,
            factorization='tucker',
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
        """Forward for 2d case."""
        batchsize, _, height, width = x.shape

        fft_dims = list(range(-self.order, 0))

        x = torch.fft.rfft2(x.float(), norm=self.fft_norm, dim=fft_dims)

        if self.order > 1:
            x = torch.fft.fftshift(x, dim=fft_dims[:-1])

        # The output will be of size (batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1)
        out_fft = torch.zeros(
            [batchsize, self.out_channels, height, width // 2 + 1],
            dtype=x.dtype,
            device=x.device,
        )

        slices0 = (
            slice(None),
            slice(None),
            slice(self.n_modes[0] // 2),
            slice(self.n_modes[1]),
        )
        slices1 = (
            slice(None),
            slice(None),
            slice(-self.n_modes[0] // 2, None),
            slice(self.n_modes[1]),
        )

        # Upper block (truncate high frequencies).
        out_fft[slices0] = self._contract(x[slices0], self.weight[slices1].to(x.device))

        # Lower block
        out_fft[slices1] = self._contract(x[slices1], self.weight[slices0].to(x.device))

        if self.order > 1:
            out_fft = torch.fft.fftshift(out_fft, dim=fft_dims[:-1])

        x = torch.fft.irfft2(out_fft, s=(height, width), dim=(-2, -1), norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias

        return x
