import numpy as np
import torch

__all__ = [
    'dht2d',
    'flip_periodic',
]


def dht2d(x: torch.Tensor, is_inverse: bool = False) -> torch.Tensor:
    if not is_inverse:
        x_ft = torch.fft.fftshift(torch.fft.fft2(x, norm='backward'), dim=(-2, -1))
    else:
        x_ft = torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm='backward')

    x_ht = x_ft.real - x_ft.imag

    if is_inverse:
        n = x.size()[-2:].numel()
        x_ht = x_ht / n

    return x_ht


def flip_periodic(x: torch.Tensor, axes: int | tuple | None = None) -> torch.Tensor:
    if axes is None:
        axes = (-2, -1)

    if isinstance(axes, int):
        axes = (axes,)

    return torch.roll(torch.flip(x, axes), shifts=(1,) * len(axes), dims=axes)
