from torch import nn

__all__ = [
    'count_parameters',
]


def count_parameters(model: nn.Module) -> int:
    return sum([p.numel() * 2 if p.is_complex() else p.numel() for p in model.parameters()])
