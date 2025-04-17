from collections import UserDict

from torch.utils.data import DataLoader

from config import BSDDatasetConfig, FNODatasetConfig

from .bsd_datasets import BSD300Synthetic
from .fno_dataset import FNODataset
from .utils import collate_wrapper

__all__ = [
    'DatasetRegistry',
]


class DatasetRegistry(UserDict):
    def load(
        self,
        data_cfgs: dict[str, FNODatasetConfig | BSDDatasetConfig],
        verbose: bool = False,
    ) -> None:
        for name, cfg in data_cfgs.items():
            dset_cls = FNODataset if isinstance(cfg, FNODatasetConfig) else BSD300Synthetic
            self.data[name] = dset_cls(**cfg.model_dump())

            if verbose:
                dset = self.data[name]
                print(  # noqa: T201
                    f'Got n_samples = {len(dset):<5} in dataset {name:<19} with sample size = {dset[0]["x"].size()}',  # noqa: E501
                )

    def make_dl(self, name: str, batch_size: int = 128, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            self.data[name],
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_wrapper,
            pin_memory=True,
        )

    def __repr__(self) -> str:
        return f'DatasetRegistry({list(self.keys())})'
