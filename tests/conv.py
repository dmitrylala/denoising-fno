import pytest
import torch

from denoising.models.conv import SpectralConv2D


@pytest.mark.parametrize('modes', [10, 16, 32])
def test_spectral_conv(modes: int) -> None:
    batch_size = 4
    s = 100
    n_dim = 2
    size = (s,) * n_dim

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    model = SpectralConv2D(
        in_channels=3,
        out_channels=1,
        n_modes=(modes, modes),
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
