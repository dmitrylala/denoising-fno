import numpy as np
import torch


def flip(
    x: np.ndarray | torch.Tensor,
    axes: int | tuple | None = None,
) -> np.ndarray | torch.Tensor:
    """
    Flips given ndarray along specified axis, while maintaining periodicity.

    Args:
        x (np.ndarray | torch.Tensor): Given tensor.
        axes (int | tuple | None, optional): Axes along which to perform the flip, along. Defaults to None.

    Returns:
        np.ndarray | torch.Tensor: Flipped tensor.

    """  # noqa: E501
    if axes is None:
        axes = (0, 1) if isinstance(x, np.ndarray) else (-2, -1)

    if isinstance(axes, int):
        axes = (axes,)

    if isinstance(x, torch.Tensor):
        return torch.roll(torch.flip(x, axes), shifts=(1,) * len(axes), dims=axes)

    flipped = np.copy(x)
    for ax in axes:
        flipped = np.roll(np.flip(flipped, axis=ax), shift=1, axis=ax)
    return flipped


def fft2d(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(x, np.ndarray):
        return np.fft.fftshift(np.fft.fft2(x))
    return torch.fft.fftshift(torch.fft.fft2(x))


def ifft2d(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(x, np.ndarray):
        return np.fft.ifft2(np.fft.ifftshift(x)).real
    return torch.fft.ifft2(torch.fft.ifftshift(x)).real


def conv2d_fft(
    x: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    return ifft2d(fft2d(x) * fft2d(y))


def sdht2d(x: np.ndarray | torch.Tensor, norm: str = 'backward') -> np.ndarray | torch.Tensor:
    """
    Separable Discrete Hartley Transform. Calculated using the Discrete Fourier Transform.

    Args:
        x (np.ndarray | torch.Tensor): Given tensor.
            For numpy: assume first two axes are spatial axes.
            For torch: assume last two axes are spatial axes.
        norm: str
            Norm type for torch fft

    Returns:
        np.ndarray | torch.Tensor: SHDT(x)

    """
    if isinstance(x, np.ndarray):
        fft = np.fft.fft2(x, axes=(0, 1))
        fft_flipped_y = flip(fft, 1)
    elif isinstance(x, torch.Tensor):
        fft = torch.fft.fft2(x, norm=norm)
        fft_flipped_y = flip(fft, -1)
    return fft_flipped_y.real - fft.imag


def isdht2d(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Inverse Separable Discrete Hartley Transform.

    Args:
        x (np.ndarray | torch.Tensor): Given array.
            For numpy: assume first two axes are spatial axes.
            For torch: assume last two axes are spatial axes.

    Returns:
        np.ndarray | torch.Tensor: SDHT^-1(x)

    """
    if isinstance(x, torch.Tensor):
        return sdht2d(x, norm='forward')
    n = np.prod(x.shape[:2])
    return sdht2d(x) / n


def even(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    return (x + flip(x)) / 2


def odd(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    return (x - flip(x)) / 2


def conv2d_sdht(
    x: np.ndarray | torch.Tensor,
    w: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    w /= w.sum()

    x_dht = sdht2d(x)
    w_dht = sdht2d(w)

    axes = 1 if isinstance(x, np.ndarray) else -1
    x_dht_flipy = flip(x_dht, axes=axes)
    w_dht_flipy = flip(w_dht, axes=axes)

    z_dht = (
        even(x_dht) * even(w_dht)
        - odd(x_dht_flipy) * odd(w_dht_flipy)
        + even(x_dht_flipy) * odd(w_dht)
        + odd(x_dht) * even(w_dht_flipy)
    )

    return isdht2d(z_dht)


def dht2d(x: np.ndarray | torch.Tensor, norm: str = 'backward') -> np.ndarray | torch.Tensor:
    if isinstance(x, np.ndarray):
        fft = np.fft.fft2(x, axes=(0, 1))
    elif isinstance(x, torch.Tensor):
        fft = torch.fft.fft2(x, norm=norm)
    return fft.real - fft.imag


def idht2d(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(x, torch.Tensor):
        return dht2d(x, norm='forward')
    n = np.prod(x.shape[:2])
    return dht2d(x) / n


def conv2d_dht(
    x: np.ndarray | torch.Tensor,
    w: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    w /= w.sum()

    x_dht = dht2d(x)
    w_dht = dht2d(w)

    return idht2d(even(x_dht) * w_dht + odd(x_dht) * flip(w_dht))
