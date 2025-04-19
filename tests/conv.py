from pathlib import Path

import pytest
import torch
from neuralop.layers.spectral_convolution import SpectralConv as SpecGt

from denoising.models.conv import FourierSpectralConv, HartleySpectralConv


@pytest.mark.parametrize('in_channels', [1, 3])
@pytest.mark.parametrize('out_channels', [1, 3])
@pytest.mark.parametrize('modes', [2, 32])
@pytest.mark.parametrize('s', [50, 100])
@pytest.mark.parametrize('factorization', ['tucker', 'dense'])
def test_fourier_conv(
    in_channels: int,
    out_channels: int,
    modes: int,
    s: int,
    factorization: str,
) -> None:
    batch_size = 4
    n_dim = 2
    size = (s,) * n_dim

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    common_kwargs = {
        'in_channels': in_channels,
        'out_channels': out_channels,
        'n_modes': (modes, modes),
        'factorization': factorization,
    }

    _ = torch.manual_seed(42)
    model = FourierSpectralConv(
        **common_kwargs,
    ).to(device)

    _ = torch.manual_seed(42)
    gt_model = SpecGt(
        **common_kwargs,
        implementation='factorized',
        device=device,
    ).to(device)

    _ = torch.manual_seed(42)
    in_data = torch.randn(batch_size, in_channels, *size, dtype=torch.float32).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, out_channels, *size]

    # Check backward pass
    loss = out.sum()
    loss.backward()

    n_unused_params = 0
    for param in model.parameters():
        if param.grad is None:
            n_unused_params += 1
    assert n_unused_params == 0, f'{n_unused_params} parameters were unused!'

    # Compare with neuralop output
    gt_out = gt_model(in_data)
    torch.testing.assert_close(out, gt_out)


@pytest.mark.parametrize(
    'in_channels',
    [
        1,
        3,
    ],
)
@pytest.mark.parametrize(
    'out_channels',
    [
        1,
        3,
    ],
)
@pytest.mark.parametrize(
    'modes',
    [
        2,
        16,
    ],
)
@pytest.mark.parametrize(
    's',
    [
        50,
        100,
    ],
)
def test_hartley_conv(
    in_channels: int,
    out_channels: int,
    modes: int,
    s: int,
) -> None:
    tests_data_dir = Path.cwd() / 'tests/data'

    batch_size = 4
    n_dim = 2
    size = (s,) * n_dim

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    common_kwargs = {
        'in_channels': in_channels,
        'out_channels': out_channels,
        'n_modes': (modes, modes),
    }

    params = f'{in_channels}-{out_channels}-{modes}-{s}'
    in_data = torch.load(tests_data_dir / f'hartley-in-{params}.pt').to(device)
    gt_out = torch.load(tests_data_dir / f'hartley-out-{params}.pt').to(device)

    _ = torch.manual_seed(42)
    model = HartleySpectralConv(
        **common_kwargs,
    ).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, out_channels, *size]

    # Check backward pass
    loss = out.sum()
    loss.backward()

    n_unused_params = 0
    for param in model.parameters():
        if param.grad is None:
            n_unused_params += 1
    assert n_unused_params == 0, f'{n_unused_params} parameters were unused!'

    # Compare with gt output
    torch.testing.assert_close(out, gt_out)
