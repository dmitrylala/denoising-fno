import tensorly as tl
import torch
from tensorly.plugins import use_opt_einsum
from tltorch.factorized_tensors.core import FactorizedTensor
from tltorch.factorized_tensors.factorized_tensors import TuckerTensor
from torch import nn

tl.set_backend('pytorch')
use_opt_einsum('optimal')
EINSUM_SYMBOLS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

__all__ = [
    'HartleySpectralConv',
    'SpectralConv',
]


def contract_tucker(x: torch.Tensor, tucker_weight: TuckerTensor) -> torch.Tensor:
    order = tl.ndim(x)

    x_syms = str(EINSUM_SYMBOLS[:order])
    out_sym = EINSUM_SYMBOLS[order]
    out_syms = list(x_syms)

    core_syms = EINSUM_SYMBOLS[order + 1 : 2 * order + 1]
    out_syms[1] = out_sym
    factor_syms = [
        EINSUM_SYMBOLS[1] + core_syms[0],
        out_sym + core_syms[1],
    ]  # out, in
    # x, y, ...
    factor_syms += [xs + rs for (xs, rs) in zip(x_syms[2:], core_syms[2:], strict=False)]

    eq = f'{x_syms},{core_syms},{",".join(factor_syms)}->{"".join(out_syms)}'
    return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)


def contract_dense(x: torch.Tensor, w: nn.Parameter) -> torch.Tensor:
    # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
    return tl.einsum('bixy,ioxy->boxy', x, w)


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
    factorization: str, 'dense' or 'tucker'
        Dense applies no factorization
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
        rank: float = 0.5,
        fft_norm: str = 'forward',
        bias: bool = True,
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

        contracts = {
            'tucker': contract_tucker,
            'dense': contract_dense,
        }
        if factorization not in contracts:
            msg = f'Unknown factorization: {factorization}'
            raise ValueError(msg)
        self._contract = contracts[factorization]

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


def flip(x: torch.Tensor, axes: int | tuple[int]) -> torch.Tensor:
    if isinstance(axes, int):
        axes = (axes,)

    flipped = x.clone()
    for ax in axes:
        flipped = torch.roll(torch.flip(x, dims=(ax,)), shifts=(1,), dims=(ax,))
    return flipped


def sdht2d(
    x: torch.Tensor,
    norm: str,
    dim: tuple[int],
    s: tuple[int] | None = None,
    inv: bool = False,
) -> torch.Tensor:
    if inv:
        fft = torch.fft.fft2(x.float(), norm=norm, dim=dim, s=s)
    else:
        fft = torch.fft.rfft2(x.float(), norm=norm, dim=dim, s=s)
    fft_flipped_y = flip(fft, axes=3)
    return fft_flipped_y.real - fft.imag


def even(x: torch.Tensor) -> torch.Tensor:
    flipped = flip(x, axes=(2, 3))
    return (x + flipped) / 2


def odd(x: torch.Tensor) -> torch.Tensor:
    flipped = flip(x, axes=(2, 3))
    return (x - flipped) / 2


def isdht2d(x: torch.Tensor, s: tuple[int], norm: str, dim: tuple[int]) -> torch.Tensor:
    n = x.size()[-2:].numel()
    x_dht = sdht2d(x, norm=norm, dim=dim, s=s, inv=True)
    return 1.0 / n * x_dht


class HartleySpectralConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int | tuple[int],
        bias: bool = True,
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
        self.weights1 = nn.Parameter(
            torch.empty(*weight_shape, dtype=torch.float).normal_(std=init_std),
        )
        self.weights2 = nn.Parameter(
            torch.randn(*weight_shape, dtype=torch.float).normal_(std=init_std),
        )

        self._contract = contract_dense

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

    def mul(self, x: torch.Tensor, w: nn.Parameter) -> torch.Tensor:
        x_dht_flipy = flip(x, axes=3)
        w_dht_flipy = flip(w, axes=3)

        return (
            self._contract(even(x), even(w))
            - self._contract(odd(x_dht_flipy), odd(w_dht_flipy))
            + self._contract(even(x_dht_flipy), odd(w))
            + self._contract(odd(x), even(w_dht_flipy))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward for 2d case."""
        batchsize, _, height, width = x.shape

        dht_dims = list(range(-self.order, 0))

        # compute x_dht
        x = sdht2d(x, norm=self.fft_norm, dim=dht_dims)

        # The output will be of size (batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1)
        out_dht = torch.zeros(
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
        out_dht[slices0] = self.mul(x[slices0], self.weights1[slices1].to(x.device))

        # Lower block
        out_dht[slices1] = self.mul(x[slices1], self.weights2[slices0].to(x.device))

        x = isdht2d(out_dht, s=(height, width), norm=self.fft_norm, dim=dht_dims)

        if self.bias is not None:
            x = x + self.bias

        return x
