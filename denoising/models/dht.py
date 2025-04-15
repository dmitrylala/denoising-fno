import torch

__all__ = [
    'even',
    'flip',
    'isdht2d',
    'odd',
    'sdht2d',
]


def flip(x: torch.Tensor, axes: int | tuple[int]) -> torch.Tensor:
    if isinstance(axes, int):
        axes = (axes,)

    flipped = x
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


def dht2d(
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
    return fft.real - fft.imag


def even(x: torch.Tensor) -> torch.Tensor:
    flipped = flip(x, axes=(-2, -1))
    return (x + flipped) / 2


def odd(x: torch.Tensor) -> torch.Tensor:
    flipped = flip(x, axes=(-2, -1))
    return (x - flipped) / 2


def isdht2d(x: torch.Tensor, s: tuple[int], norm: str, dim: tuple[int]) -> torch.Tensor:
    n = x.size()[-2:].numel()
    x_dht = sdht2d(x, norm=norm, dim=dim, s=s, inv=True)
    return 1.0 / n * x_dht


def idht2d(x: torch.Tensor, s: tuple[int], norm: str, dim: tuple[int]) -> torch.Tensor:
    n = x.size()[-2:].numel()
    x_dht = dht2d(x, norm=norm, dim=dim, s=s, inv=True)
    return 1.0 / n * x_dht
