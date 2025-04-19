import numpy as np
import torch

BACKEND_NP = 'numpy'
BACKEND_TORCH = 'torch'


def make_impulse(size: int, backend: str = BACKEND_NP) -> tuple:
    x = np.ones((size, size))
    x_dht = np.zeros((size, size))
    x_dht[0, 0] = size**2
    if backend == BACKEND_TORCH:
        x, x_dht = torch.tensor(x), torch.tensor(x_dht)
    return x, x_dht


def make_cos2d(size: int, k: int, backend: str = BACKEND_NP) -> tuple:
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

    if backend == BACKEND_TORCH:
        x, x_dht = torch.tensor(x), torch.tensor(x_dht)
    return x, x_dht
