import numpy as np


def fft2d(c: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(c))


def conv2d_fft(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    convolved_fft = fft2d(x) * fft2d(y)
    return np.fft.ifft2(np.fft.ifftshift(convolved_fft)).real


def make_impulse(size: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.ones((size, size))
    x_dht = np.zeros((size, size))
    x_dht[0, 0] = size**2
    return x, x_dht


def make_cos2d(size: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    c = 2.0 * np.pi * k / size
    i = np.arange(size)
    x = np.cos(c * i)
    x = np.outer(x, x)

    x_dht = np.zeros((size, size))
    val = (size / 2.0) ** 2
    x_dht[k][k] += val
    x_dht[k][-k] += val
    x_dht[-k][-k] += val
    x_dht[-k][k] += val

    return x, x_dht
