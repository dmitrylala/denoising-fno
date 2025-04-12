import torch
from torch import nn

__all__ = [
    'SoftGating',
]


class SoftGating(nn.Module):
    """
    Applies soft-gating by weighting the channels of the given input.

    Given an input x of size `(batch-size, channels, height, width)`,
    this returns `x * w `
    where w is of shape `(1, channels, 1, 1)`

    Parameters
    ----------
    in_features : int
    out_features : None
        this is provided for API compatibility with nn.Linear only
    n_dim : int, default is 2
        Dimensionality of the input (excluding batch-size and channels).
        ``n_dim=2`` corresponds to having Module2D.
    bias : bool, default is False

    """

    def __init__(
        self,
        in_features: int,
        out_features: int | None = None,
        n_dim: int = 2,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if out_features is not None and in_features != out_features:
            msg = f'Got in_features={in_features} and out_features={out_features}, but these two must be the same for soft-gating'  # noqa: E501
            raise ValueError(msg)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))
        if bias:
            self.bias = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies soft-gating to a batch of activations."""
        if self.bias is not None:
            return self.weight * x + self.bias
        return self.weight * x
