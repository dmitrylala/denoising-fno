from abc import abstractmethod

import numpy as np
import tensorly as tl
import torch
from tensorly.plugins import use_opt_einsum
from tltorch.factorized_tensors.core import FactorizedTensor
from tltorch.factorized_tensors.factorized_tensors import TuckerTensor
from torch import nn

from .dht import (
    dht2d,
    even,
    flip,
    flip_new,
    idht2d,
    isdht2d,
    odd,
    sdht2d,
)

tl.set_backend('pytorch')
use_opt_einsum('optimal')
EINSUM_SYMBOLS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

__all__ = [
    'FourierSpectralConv',
    'HartleySeparableSpectralConv',
    'HartleySpectralConv',
    'HartleySpectralConvV4',
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


class BaseSpectralConv(nn.Module):
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
        by default 'forward'

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
        dtype: torch.dtype = torch.cfloat,
        apply_shift: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.factorization = factorization

        # n_modes is the total number of modes kept along each dimension
        self.n_modes = n_modes
        self.order = len(self.n_modes)
        self.apply_shift = apply_shift

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
            dtype=dtype,
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

    @abstractmethod
    def forward_spectral(self, x: torch.Tensor, dims: tuple[int]) -> torch.Tensor:
        pass

    @abstractmethod
    def conv_spectral(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def inverse_spectral(
        self,
        x: torch.Tensor,
        sizes: tuple[int],
        dims: tuple[int],
    ) -> torch.Tensor:
        pass

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

        x = self.forward_spectral(x, dims=fft_dims)

        # if self.apply_shift and self.order > 1:
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

        out_fft[slices_x] = self.conv_spectral(x[slices_x], weight)

        # if self.apply_shift and self.order > 1:
        out_fft = torch.fft.ifftshift(out_fft, dim=fft_dims[:-1])

        x = self.inverse_spectral(out_fft, sizes=mode_sizes, dims=fft_dims)

        if self.bias is not None:
            x = x + self.bias

        return x


class FourierSpectralConv(BaseSpectralConv):
    def forward_spectral(self, x: torch.Tensor, dims: tuple[int]) -> torch.Tensor:
        return torch.fft.rfftn(x, norm=self.fft_norm, dim=dims)

    def conv_spectral(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return self._contract(x, w)

    def inverse_spectral(
        self,
        x: torch.Tensor,
        sizes: tuple[int],
        dims: tuple[int],
    ) -> torch.Tensor:
        return torch.fft.irfftn(x, s=sizes, dim=dims, norm=self.fft_norm)


class HartleySeparableSpectralConv(BaseSpectralConv):
    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        kwargs.update({'apply_shift': False, 'dtype': torch.float, 'factorization': 'dense'})
        super().__init__(**kwargs)

    def forward_spectral(self, x: torch.Tensor, dims: tuple[int]) -> torch.Tensor:
        return sdht2d(x, norm=self.fft_norm, dim=dims)

    def conv_spectral(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x_dht_flipy = flip(x, axes=3)
        w_dht_flipy = flip(w, axes=3)

        return (
            self._contract(even(x), even(w))
            - self._contract(odd(x_dht_flipy), odd(w_dht_flipy))
            + self._contract(even(x_dht_flipy), odd(w))
            + self._contract(odd(x), even(w_dht_flipy))
        )

    def inverse_spectral(
        self,
        x: torch.Tensor,
        sizes: tuple[int],
        dims: tuple[int],
    ) -> torch.Tensor:
        return isdht2d(x, s=sizes, dim=dims, norm=self.fft_norm)


class HartleySpectralConv(BaseSpectralConv):
    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        kwargs.update({'apply_shift': False, 'dtype': torch.float, 'factorization': 'dense'})
        super().__init__(**kwargs)

    def forward_spectral(self, x: torch.Tensor, dims: tuple[int]) -> torch.Tensor:
        return dht2d(x, norm=self.fft_norm, dim=dims)

    # R_H = P_HE * Q_H + P_HO * Q_H(-u,-v)  # noqa: ERA001
    def conv_spectral(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x_even = even(x)
        x_odd = odd(x)
        return self._contract(x_even, w) + self._contract(x_odd, flip(w, (-2, -1)))

    def inverse_spectral(
        self,
        x: torch.Tensor,
        sizes: tuple[int],
        dims: tuple[int],
    ) -> torch.Tensor:
        return idht2d(x, s=sizes, dim=dims, norm=self.fft_norm)


def dht2d_new(x: torch.Tensor, is_inverse: bool = False) -> torch.Tensor:
    x_ft = torch.fft.fft2(x, norm='backward')  # norm='backward' applies no normalization
    x_ht = x_ft.real - x_ft.imag

    if is_inverse:
        n = x.size()[-2:].numel()
        x_ht = x_ht / n

    return x_ht


class HartleySpectralConvV4(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int | tuple[int],
        factorization: str = 'dense',
        fft_norm: str = 'forward',
        bias: bool = True,
        dtype: torch.dtype = torch.float,
        **_,  # noqa: ANN003
    ) -> None:
        super().__init__()

        if factorization != 'dense':
            msg = 'Supported only dense weight tensors'
            raise ValueError(msg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factorization = factorization

        # n_modes is the total number of modes kept along each dimension
        self.n_modes = n_modes
        self.order = len(self.n_modes)

        self.fft_norm = fft_norm

        # Create spectral weight tensor
        init_std = (2 / (in_channels + out_channels)) ** 0.5
        weight_shape = (in_channels, out_channels, *tuple(np.array(self.n_modes) * 2))
        self.weight = FactorizedTensor.new(
            weight_shape,
            factorization=factorization,
            fixed_rank_modes=None,
            dtype=dtype,
        )
        self.weight.normal_(0, init_std)

        # Contraction function
        self._contract = contract_dense

        self.bias = None
        if bias:
            self.bias = nn.Parameter(
                init_std * torch.randn(*((self.out_channels,) + (1,) * self.order)),
            )

    def hartley_conv(
        self,
        x: torch.Tensor,
        x_reverse: torch.Tensor,
        kernel: torch.Tensor,
        kernel_reverse: torch.Tensor,
    ) -> torch.Tensor:
        x_even = (x + x_reverse) / 2
        x_odd = (x - x_reverse) / 2
        return self._contract(x_even, kernel) + self._contract(x_odd, kernel_reverse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, h, w = x.size()
        modes_h, modes_w = self.n_modes
        if h < 2 * modes_h or w < 2 * modes_w:
            msg = f'Expected input with bigger spatial dims: h>={w * modes_h}, w>={2 * modes_w}, got: {h=}, {w=}'  # noqa: E501
            raise ValueError(msg)

        # a = torch.fft.fftshift(torch.fft.fft2(x, norm='backward'))
        # x_shifted = a.real - a.imag

        x = dht2d_new(x)

        kernel = self.weight
        kernel_reverse = flip_new(kernel)
        x_reverse = flip_new(x)

        slices_bc = [slice(None), slice(None)]

        slices = [*slices_bc, slice(None, modes_h), slice(None, modes_w)]
        left_upper = self.hartley_conv(
            x[slices],
            x_reverse[slices],
            kernel[slices],
            kernel_reverse[slices],
        )

        # c = tuple(s // 2 - 1 for s in x.size()[-2:])
        # x_shifted_reverse = flip_new(x_shifted)
        # slices_x =  [*slices_bc, slice(None, modes_h), slice(-modes_w, None)]
        # slices_w =  [*slices_bc, slice(None, modes_h), slice(None, modes_w)]
        # left_upper = self.hartley_conv(
        #     x_shifted[slices_x], x_shifted_reverse[slices_x],
        #     kernel[slices_w], kernel_reverse[slices_w],
        # )

        slices = [*slices_bc, slice(None, modes_h), slice(-modes_w, None)]
        right_upper = self.hartley_conv(
            x[slices],
            x_reverse[slices],
            kernel[slices],
            kernel_reverse[slices],
        )

        slices = [*slices_bc, slice(-modes_h, None), slice(None, modes_w)]
        left_down = self.hartley_conv(
            x[slices],
            x_reverse[slices],
            kernel[slices],
            kernel_reverse[slices],
        )

        slices = [*slices_bc, slice(-modes_h, None), slice(-modes_w, None)]
        right_down = self.hartley_conv(
            x[slices],
            x_reverse[slices],
            kernel[slices],
            kernel_reverse[slices],
        )

        # concat corners along y axis with zero padding between
        pad_shape = [batch_size, self.out_channels, modes_h, w - 2 * modes_w]
        pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        upper = torch.concat([left_upper, pad_zeros, right_upper], axis=-1)
        down = torch.concat([left_down, pad_zeros, right_down], axis=-1)

        # concat upper and down fragments along x axis with zero padding between
        pad_shape = [batch_size, self.out_channels, h - 2 * modes_h, w]
        pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x = torch.concat([upper, pad_zeros, down], axis=-2)

        x = dht2d_new(x, is_inverse=True)

        if self.bias is not None:
            x = x + self.bias

        return x
