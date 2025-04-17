from collections import UserDict

import torch
from neuralop.models.fno import TFNO
from torch import nn

from config import ModelConfig

from .fno import FNO
from .utils import count_parameters

__all__ = [
    'ModelRegistry',
]


class ModelRegistry(UserDict):
    def load(
        self,
        models_cfgs: dict[str, ModelConfig],
        random_seed: int = 42,
        device: str | torch.device = 'cpu',
        verbose: bool = False,
    ) -> None:
        for model_name, cfg in models_cfgs.items():
            self.data[model_name] = self._load_model(
                cfg,
                random_seed=random_seed,
                device=device,
            )
            if verbose:
                op_done = 'Loaded' if cfg.weights_path else 'Created'
                print(  # noqa: T201
                    f'{op_done:<7} model {model_name:<16} with n_parameters = {count_parameters(self.data[model_name])}',  # noqa: E501
                )

    @staticmethod
    def _load_model(
        cfg: ModelConfig,
        random_seed: int = 42,
        device: str | torch.device = 'cpu',
    ) -> nn.Module:
        _ = torch.manual_seed(random_seed)
        model_class = resolve_model_cls(cfg.model_class_name)
        model = model_class(**cfg.fno_cfg)
        if cfg.weights_path:
            model.load_state_dict(
                torch.load(str(cfg.weights_path), weights_only=False, map_location=device),
            )
            _ = model.eval()
        return model.to(device)

    def __repr__(self) -> str:
        return f'ModelRegistry({list(self.keys())})'


def resolve_model_cls(cls_name: str) -> type:
    if cls_name == 'FNO':
        model_cls = FNO
    elif cls_name == 'TFNO':
        model_cls = TFNO
    else:
        msg = f'Got unknown model type: {cls_name}'
        raise ValueError(msg)
    return model_cls
