from pathlib import Path

import numpy as np
import torch

from .file_dataset import FileDataset

__all__ = [
    'FNODataset',
]


class FNODataset(FileDataset):
    def __init__(
        self,
        root: str | Path,
        sample_list: str | Path,
        load_params: list[dict],
        transforms: list | None = None,
        normalize: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            sample_list=sample_list,
            load_params=load_params,
            transforms=transforms,
        )
        self.normalize = normalize

    def __getitem__(self, index: int | slice) -> dict[str, torch.Tensor]:
        x, y = super().__getitem__(index)

        def to_tensor(img: np.ndarray) -> torch.Tensor:
            img = torch.from_numpy(img).clone().type(torch.float32)
            if self.normalize:
                img /= 255.0

            # add channel_dim, if it's not present, for grayscale for example
            if len(img.shape) == 2:  # noqa: PLR2004
                img = img.unsqueeze(-1)

            # convert to CHW
            return img.permute(2, 0, 1)

        return {'x': to_tensor(x), 'y': to_tensor(y)}
