from pathlib import Path

from torchmetrics.metric import Metric

from .configs import (
    BSDDatasetConfig,
    FNODatasetConfig,
    LoadParams,
    MetricConfig,
    ModelConfig,
    TrainerConfig,
    WandbConfig,
)

__all__ = [
    'make_bsd_dset_config',
    'make_fno_dset_config',
    'make_load_params',
    'make_metric_config',
    'make_model_config',
    'make_trainer_config',
    'make_wandb_config',
]


def make_model_config(fno_cfg: dict, weights_path: Path, model_class_name: str) -> ModelConfig:
    return ModelConfig(
        fno_cfg=fno_cfg, weights_path=weights_path, model_class_name=model_class_name
    )


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


def make_wandb_config(run_name: str, tags: list[str] | None = None) -> WandbConfig:
    wandb_tags = [] if tags is None else tags
    return WandbConfig(name=run_name, tags=wandb_tags)


def make_trainer_config(
    n_epochs: int,
    lr: float,
    wandb_log: bool,
    save_dir: str | Path,
    verbose: bool,
) -> TrainerConfig:
    return TrainerConfig(
        n_epochs=n_epochs,
        lr=lr,
        wandb_log=wandb_log,
        log_output=True,
        eval_interval=1,
        save_every=1,
        save_dir=Path(save_dir),
        verbose=verbose,
    )
