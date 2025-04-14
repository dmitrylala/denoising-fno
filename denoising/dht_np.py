import numpy as np


def flip(x: np.ndarray, axes: int | tuple = 0) -> np.ndarray:
    """
    Flips given ndarray along specified axis, while maintaining periodicity.

    Args:
        x (np.ndarray): Given ndarray.
        axes (int | tuple, optional): Axes along which to perform the flip, along. Defaults to 0.

    Returns:
        np.ndarray: Flipped ndarray.

    """
    if isinstance(axes, int):
        axes = (axes,)

    flipped = np.copy(x)
    for ax in axes:
        flipped = np.roll(np.flip(flipped, axis=ax), shift=1, axis=ax)
    return flipped


def sdht2d(x: np.ndarray) -> np.ndarray:
    """
    Separable Discrete Hartley Transform. Calculated using the Discrete Fourier Transform.

    Args:
        x (np.ndarray): Given ndarray. Assume first two axes are spatial axes.

    Returns:
        np.ndarray: SHDT(x)

    """
    fft = np.fft.fft2(x, axes=(0, 1))
    fft_flipped_y = flip(fft, axes=1)
    return fft_flipped_y.real - fft.imag


def isdht2d(x: np.ndarray) -> np.ndarray:
    """
    Inverse Separable Discrete Hartley Transform.

    Args:
        x (np.ndarray): Given ndarray. Assume first two axes are spatial axes.

    Returns:
        np.ndarray: SDHT^-1(x)

    """
    n = np.prod(x.shape[:2])
    x_dht = sdht2d(x)
    return 1.0 / n * x_dht


def even(x: np.ndarray) -> np.ndarray:
    flipped = flip(x, axes=(0, 1))
    return (x + flipped) / 2


def odd(x: np.ndarray) -> np.ndarray:
    flipped = flip(x, axes=(0, 1))
    return (x - flipped) / 2


def conv2d_dht(x_src: np.ndarray, w: np.ndarray, padding: str | None = None) -> np.ndarray:
    x = np.copy(x_src)

    # apply padding
    if padding is not None:
        pad_width = [(w.shape[0], w.shape[0]), (w.shape[1], w.shape[1])] + [(0, 0)] * (x.ndim - 2)
        x = np.pad(x, pad_width, mode=padding)

    w_large = np.zeros_like(x)
    w_large[: w.shape[0], : w.shape[1], ...] = w

    x_dht = sdht2d(x)
    w_dht = sdht2d(w_large)

    x_dht_flipy = flip(x_dht, axes=1)
    w_dht_flipy = flip(w_dht, axes=1)

    z_dht = (
        even(x_dht) * even(w_dht)
        - odd(x_dht_flipy) * odd(w_dht_flipy)
        + even(x_dht_flipy) * odd(w_dht)
        + odd(x_dht) * even(w_dht_flipy)
    )
    z = isdht2d(z_dht)

    # pad back
    if padding is not None:
        slices = []
        for p1, p2 in pad_width:
            if p1 == p2 == 0:
                slices.append(slice(None))
                continue
            slices.append(slice(p1, -p2))
        z = z[tuple(slices)]

    return z
