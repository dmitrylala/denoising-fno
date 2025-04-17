from collections import defaultdict

import torch
from torchmetrics.metric import Metric

from config import MetricConfig

__all__ = [
    'Metrics',
]


class Metrics:
    def __init__(self, cfgs: list[MetricConfig]) -> None:
        self.cfgs = cfgs
        self.metrics = {}

    def create(self, evaluator_names: list[str]) -> None:
        self.metrics = defaultdict(lambda: defaultdict(Metric))
        for name in evaluator_names:
            for m_cfg in self.cfgs:
                self.metrics[name][m_cfg.name] = m_cfg.cls(**m_cfg.kwargs)

    def update(self, name: str, preds: torch.Tensor, gt: torch.Tensor) -> None:
        for metric_name in self.metrics[name]:
            self.metrics[name][metric_name].update(preds, gt)

    def compute(self) -> dict[str, float]:
        values = {}
        for name in self.metrics:
            for metric_name in self.metrics[name]:
                values[f'{name}_{metric_name}'] = self.metrics[name][metric_name].compute()
        return values

    def __repr__(self) -> str:
        return f'Metrics({[cfg.name for cfg in self.cfgs]})'
