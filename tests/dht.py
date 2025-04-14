import inspect
from collections.abc import Callable

import numpy as np
import pytest

from denoising.dht_np import conv2d_dht, isdht2d, sdht2d

from .utils import conv2d_fft, make_cos2d, make_impulse


def test_dht2d_constant() -> None:
    # DHT transform of a constant sequence, based on DFT transform of the same
    size = 20
    x, x_dht_gt = make_impulse(size)
    x_dht = sdht2d(x)
    msg = f'{inspect.stack()[0][3]}: constant sequence test failed'
    np.testing.assert_allclose(x_dht, x_dht_gt, atol=1e-08, err_msg=msg)


# 0 and 19 are corner-cases for size=20
@pytest.mark.parametrize('k', [3, 0, 19])
def test_dht2d_cosine(k: int) -> None:
    # DHT transform of a cosine, based on DFT transform of the same
    size = 20
    x, x_dht_gt = make_cos2d(size, k)
    x_dht = sdht2d(x)
    msg = f'{inspect.stack()[0][3]}: cosine sequence test failed'
    np.testing.assert_allclose(x_dht, x_dht_gt, atol=1e-08, err_msg=msg)


def test_idht2d_impulse() -> None:
    # IDHT transform of an impulse at 0, based on IDFT transform of the same
    size = 20
    x_idht_gt, x = make_impulse(size)
    x_idht = isdht2d(x)
    msg = f'{inspect.stack()[0][3]}: impulse test failed'
    np.testing.assert_allclose(x_idht, x_idht_gt, atol=1e-08, err_msg=msg)


# 0 and 19 are corner-cases for size=20
@pytest.mark.parametrize('k', [3, 0, 19])
def test_idht2d_cosine(k: int) -> None:
    # IDHT transform of sum of impulses at k and N-k, based on IDFT transform of the same
    size = 20
    x_idht_gt, x = make_cos2d(size, k)
    x_idht = isdht2d(x)
    msg = f'{inspect.stack()[0][3]}: sum of impulses test failed'
    np.testing.assert_allclose(x_idht, x_idht_gt, atol=1e-08, err_msg=msg)


@pytest.mark.parametrize(
    ('size', 'func'),
    [
        (20, lambda s: np.ones((s, s))),
        (20, lambda s: np.random.default_rng(seed=42).normal(size=(s, s))),
    ],
)
def test_conv2d_dht(size: int, func: Callable) -> None:
    x = func(size)
    y = func(size)
    z = conv2d_dht(x, y)
    z_gt = conv2d_fft(x, y)
    msg = f'{inspect.stack()[0][3]} test failed'
    np.testing.assert_allclose(z, z_gt, err_msg=msg)
