from pathlib import Path

from torch.utils.data.dataset import Dataset

from .sampler import CSVSampler
from .utils import read_img

__all__ = [
    'FileDataset',
]


class FileDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        sample_list: str | Path,
        load_params: list[dict],
        transforms: list | None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.sampler = CSVSampler(sample_list)
        self.load_func = read_img
        self.load_params = load_params
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.sampler)

    def _load(self, x: dict | list) -> dict | list:
        if isinstance(x, dict):
            res = {}
            for (k, fname), kwargs in zip(x.items(), self.load_params, strict=True):
                res[k] = self.load_func(self.root / fname, **kwargs)
            return res

        if isinstance(x, list | tuple):
            res = []
            for fname, kwargs in zip(x, self.load_params, strict=True):
                res.append(self.load_func(self.root / fname, **kwargs))
            return res

        return self.load_func(self.root / x, **self.load_params)

    def __getitem__(self, idx: int | slice) -> dict | list:
        sample = self._load(self.sampler[idx])
        if self.transforms:
            sample = self.transforms(sample)
        return sample
