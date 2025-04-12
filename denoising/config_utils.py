from pathlib import Path

from neuralop.models import TFNO
from torchmetrics.metric import Metric

from .configs import BSDDatasetConfig, FNODatasetConfig, LoadParams, MetricConfig, ModelConfig
from .models import FNO

__all__ = [
    'make_bsd_dset_config',
    'make_fno_dset_config',
    'make_load_params',
    'make_metric_config',
    'make_model_config',
]


def make_model_config(fno_cfg: dict, weights_path: Path, model_cls_name: str) -> ModelConfig:
    if model_cls_name == 'FNO':
        model_cls = FNO
    elif model_cls_name == 'TFNO':
        model_cls = TFNO
    else:
        msg = f'Got unknown model type: {model_cls_name}'
        raise ValueError(msg)
    return ModelConfig(fno_cfg=fno_cfg, weights_path=weights_path, model_class=model_cls)


def make_load_params(key: str, shape: list[int], dtype: str) -> LoadParams:
    return LoadParams(key=key, shape=shape, dtype=dtype)


def make_fno_dset_config(
    root: str,
    sample_list: str,
    load_params: list[LoadParams],
    normalize: bool = False,
) -> FNODatasetConfig:
    return FNODatasetConfig(
        root=Path(root),
        sample_list=Path(sample_list),
        load_params=load_params,
        normalize=normalize,
    )


def make_bsd_dset_config(root: str, variance: float, mode: str) -> BSDDatasetConfig:
    return BSDDatasetConfig(
        root=Path(root),
        variance=variance,
        mode=mode,
    )


def make_metric_config(name: str, metric_cls: Metric, kwargs: dict) -> MetricConfig:
    return MetricConfig(name=name, cls=metric_cls, kwargs=kwargs)
