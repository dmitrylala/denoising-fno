from torch import nn

__all__ = [
    'count_parameters',
]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
