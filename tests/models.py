import pytest
import torch

from denoising.models import FNO


@pytest.mark.parametrize('factorization', ['tucker'])
@pytest.mark.parametrize('n_dim', [1, 2, 3, 4])
@pytest.mark.parametrize('lifting_channel_ratio', [1, 2])
def test_fno(factorization: str, n_dim: int, lifting_channel_ratio: int) -> None:
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

    rank = 0.2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    model = FNO(
        in_channels=3,
        out_channels=1,
        hidden_channels=width,
        n_modes=n_modes,
        factorization=factorization,
        rank=rank,
        n_layers=n_layers,
        fc_channels=fc_channels,
        lifting_channel_ratio=lifting_channel_ratio,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size, dtype=torch.float32).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]

    # Check backward pass
    loss = out.sum()
    loss.backward()

    n_unused_params = 0
    for param in model.parameters():
        if param.grad is None:
            n_unused_params += 1
    assert n_unused_params == 0, f'{n_unused_params} parameters were unused!'
