from pathlib import Path
from typing import Annotated

import torch
from pydantic import AfterValidator, BaseModel, NonNegativeInt, field_serializer
from pydantic_settings import SettingsConfigDict
from torch import device

__all__ = [
    'TrainConfig',
    'TrainerConfig',
]


def device_exists(d: str) -> device:
    return device(d)


Device = Annotated[str, AfterValidator(device_exists)]


class TrainerConfig(BaseModel):
    n_epochs: int
    lr: float
    wandb_log: bool = False
    eval_interval: int = 1
    log_output: bool = False
    save_every: int | None = None
    save_best: int | None = None
    save_dir: Path = Path('./ckpt')
    verbose: bool = False


class TrainConfig(BaseModel):
    model_config = SettingsConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    # Datasets params
    train_dset: str
    test_dset: str
    train_batch_size: int
    test_batch_size: int

    # Model params
    name_model: str
    cfg_fno: dict
    cls_model: str = 'FNO'

    # Run params
    random_seed: NonNegativeInt
    device: Device
    run_name: str
    save_weights_path: Path | None = None

    # Train params
    n_epochs: int
    lr: float
    wandb_log: bool = False
    save_dir: Path = Path('./ckpt')
    verbose: bool = False

    @field_serializer('device')
    def serialize_device(self, device: torch.device, _) -> str:  # noqa: ANN001
        return str(device)
