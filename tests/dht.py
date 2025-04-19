import inspect
from collections.abc import Callable

import numpy as np
import pytest
import torch

from denoising.dht import (
    conv2d_dht,
    conv2d_fft,
    conv2d_sdht,
    dht2d,
    fft2d,
    idht2d,
    ifft2d,
    isdht2d,
    sdht2d,
)
from denoising.utils import seed_everything

from .utils import BACKEND_NP, BACKEND_TORCH, make_cos2d, make_impulse


@pytest.mark.parametrize('backend', [BACKEND_NP, BACKEND_TORCH])
@pytest.mark.parametrize('dht_fn', [sdht2d, dht2d])
def test_dht2d_constant(dht_fn: Callable, backend: str) -> None:
    # DHT transform of a constant sequence, based on DFT transform of the same
    size = 20
    x, x_dht_gt = make_impulse(size, backend=backend)
    x_dht = dht_fn(x)
    msg = f'{inspect.stack()[0][3]}: constant sequence test failed'
    if backend == BACKEND_NP:
        assert isinstance(x_dht, np.ndarray)
        np.testing.assert_allclose(x_dht, x_dht_gt, atol=1e-08, err_msg=msg)
    else:
        assert isinstance(x_dht, torch.Tensor)
        torch.testing.assert_close(x_dht, x_dht_gt, msg=msg)


# 0 and 19 are corner-cases for size=20
@pytest.mark.parametrize('backend', [BACKEND_NP, BACKEND_TORCH])
@pytest.mark.parametrize('k', [3, 0, 19])
@pytest.mark.parametrize('dht_fn', [sdht2d, dht2d])
def test_dht2d_cosine(dht_fn: Callable, k: int, backend: str) -> None:
    # DHT transform of a cosine, based on DFT transform of the same
    size = 20
    x, x_dht_gt = make_cos2d(size, k, backend=backend)
    x_dht = dht_fn(x)
    msg = f'{inspect.stack()[0][3]}: cosine sequence test failed'
    if backend == BACKEND_NP:
        assert isinstance(x_dht, np.ndarray)
        np.testing.assert_allclose(x_dht, x_dht_gt, atol=1e-08, err_msg=msg)
    else:
        assert isinstance(x_dht, torch.Tensor)
        torch.testing.assert_close(x_dht, x_dht_gt, msg=msg)


@pytest.mark.parametrize('backend', [BACKEND_NP, BACKEND_TORCH])
@pytest.mark.parametrize('idht_fn', [isdht2d, idht2d])
def test_idht2d_impulse(idht_fn: Callable, backend: str) -> None:
    # IDHT transform of an impulse at 0, based on IDFT transform of the same
    size = 3
    x_idht_gt, x = make_impulse(size, backend=backend)
    x_idht = idht_fn(x)
    msg = f'{inspect.stack()[0][3]}: impulse test failed'
    if backend == BACKEND_NP:
        assert isinstance(x_idht, np.ndarray)
        np.testing.assert_allclose(x_idht, x_idht_gt, atol=1e-08, err_msg=msg)
    else:
        assert isinstance(x_idht, torch.Tensor)
        torch.testing.assert_close(x_idht, x_idht_gt, msg=msg)


# 0 and 19 are corner-cases for size=20
@pytest.mark.parametrize('backend', [BACKEND_NP, BACKEND_TORCH])
@pytest.mark.parametrize('k', [3, 0, 19])
@pytest.mark.parametrize('idht_fn', [isdht2d, idht2d])
def test_idht2d_cosine(idht_fn: Callable, k: int, backend: str) -> None:
    # IDHT transform of sum of impulses at k and N-k, based on IDFT transform of the same
    size = 20
    x_idht_gt, x = make_cos2d(size, k, backend=backend)
    x_idht = idht_fn(x)
    msg = f'{inspect.stack()[0][3]}: sum of impulses test failed'
    if backend == BACKEND_NP:
        assert isinstance(x_idht, np.ndarray)
        np.testing.assert_allclose(x_idht, x_idht_gt, atol=1e-08, err_msg=msg)
    else:
        assert isinstance(x_idht, torch.Tensor)
        torch.testing.assert_close(x_idht, x_idht_gt, msg=msg)


@pytest.mark.parametrize('backend', [BACKEND_NP, BACKEND_TORCH])
@pytest.mark.parametrize(
    'size',
    [
        (3, 3),
        (10, 2),
        (2, 10),
        (20, 20),
        (100, 1),
        (1, 100),
        (100, 100),
        (1000, 1000),
    ],
)
@pytest.mark.parametrize(
    'call',
    [
        lambda x: isdht2d(sdht2d(x)),
        lambda x: idht2d(dht2d(x)),
        lambda x: ifft2d(fft2d(x)),
    ],
)
def test_inverse(size: tuple[int], call: Callable, backend: str) -> None:
    rng = np.random.default_rng(42)
    seed_everything(42)
    x = rng.normal(size=size) if backend == BACKEND_NP else torch.randn(size)

    x_out = call(x)
    msg = f'{inspect.stack()[0][3]}: {call=} failed'
    if backend == BACKEND_NP:
        assert isinstance(x_out, np.ndarray)
        np.testing.assert_allclose(x_out, x, atol=1e-14, err_msg=msg)
    else:
        assert isinstance(x_out, torch.Tensor)
        torch.testing.assert_close(x_out, x, msg=msg)


@pytest.mark.parametrize('backend', [BACKEND_NP, BACKEND_TORCH])
@pytest.mark.parametrize(
    'func_name',
    [
        'ones',
        'random',
    ],
)
@pytest.mark.parametrize(
    'conv_fn',
    [
        conv2d_sdht,
        conv2d_dht,
    ],
)
@pytest.mark.parametrize(
    ('size'),
    [
        (3, 3),
        (20, 3),
        (3, 20),
        (20, 20),
        (100, 100),
        (1000, 1000),
    ],
)
def test_conv_theorem(
    size: tuple[int],
    func_name: str,
    conv_fn: Callable,
    backend: str,
) -> None:
    seed_everything(42)
    if backend == BACKEND_NP:
        if func_name == 'ones':
            x = np.ones(size)
            y = np.ones(size)
        else:
            rng = np.random.default_rng(42)
            x = rng.normal(size=size)
            y = rng.uniform(size=size)
    elif func_name == 'ones':
        x = torch.ones(size)
        y = torch.ones(size)
    else:
        x = torch.randn(size)
        y = torch.rand(size)

    z = conv_fn(x, y)
    z_gt = conv2d_fft(x, y)
    msg = f'{inspect.stack()[0][3]} test failed'
    if backend == BACKEND_NP:
        assert isinstance(z, np.ndarray)
        np.testing.assert_allclose(z, z_gt, err_msg=msg)
    else:
        assert isinstance(z, torch.Tensor)
        torch.testing.assert_close(z, z_gt, msg=msg)
