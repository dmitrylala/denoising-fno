import pytest
import torch
from neuralop.models.fno import FNO as FNOGt  # noqa: N811

from denoising.models import FNO


@pytest.mark.parametrize('in_channels', [1, 3])
@pytest.mark.parametrize('out_channels', [1, 3])
@pytest.mark.parametrize('lifting_channel_ratio', [1, 2])
def test_fno(in_channels: int, out_channels: int, lifting_channel_ratio: int) -> None:
    if torch.cuda.is_available():
        device = 'cuda'
        s = 16
        modes = 8
        width = 16
        fc_channels = 16
        batch_size = 4
        n_layers = 4
    else:
        device = 'cpu'
        s = 16
        modes = 5
        width = 15
        fc_channels = 32
        batch_size = 3
        n_layers = 2

    n_dim = 2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    model = FNO(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=width,
        n_modes=n_modes,
        n_layers=n_layers,
        fc_channels=fc_channels,
        lifting_channel_ratio=lifting_channel_ratio,
    ).to(device)

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


@pytest.mark.parametrize('in_channels', [1, 3])
@pytest.mark.parametrize('out_channels', [1, 3])
@pytest.mark.parametrize('factorization', ['tucker', 'dense'])
def test_fno_neuralop(in_channels: int, out_channels: int, factorization: str) -> None:
    if torch.cuda.is_available():
        device = 'cuda'
        s = 16
        modes = 8
        width = 16
        fc_channels = 16
        batch_size = 4
        n_layers = 4
        lifting_channel_ratio = 2
    else:
        device = 'cpu'
        s = 16
        modes = 5
        width = 15
        fc_channels = 32
        batch_size = 3
        n_layers = 2
        lifting_channel_ratio = 1

    n_dim = 2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim

    common_kwargs = {
        'in_channels': in_channels,
        'out_channels': out_channels,
        'hidden_channels': width,
        'n_modes': n_modes,
        'n_layers': n_layers,
        'fc_channels': fc_channels,
        'lifting_channel_ratio': lifting_channel_ratio,
        'rank': 0.42,
        'factorization': factorization,
    }

    _ = torch.manual_seed(42)
    model = FNO(
        **common_kwargs,
        spectral='fourier',
    ).to(device)

    _ = torch.manual_seed(42)
    gt_model = FNOGt(
        **common_kwargs,
        device=device,
    ).to(device)

    _ = torch.manual_seed(42)
    in_data = torch.randn(batch_size, in_channels, *size, dtype=torch.float32).to(device)

    out = model(in_data)
    gt_out = gt_model(in_data)

    torch.testing.assert_close(out, gt_out)
