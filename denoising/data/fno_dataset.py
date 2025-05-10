from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import Resize

from .file_dataset import FileDataset

__all__ = [
    'FNODataset',
]


class FNODataset(FileDataset):
    def __init__(  # noqa: PLR0913
        self,
        root: str | Path,
        sample_list: str | Path,
        load_params: list[dict],
        transforms: list | None = None,
        normalize: bool = False,
        resize_y: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            sample_list=sample_list,
            load_params=load_params,
            transforms=transforms,
        )
        self.normalize = normalize
        self.resize_y = resize_y

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

        x, y = to_tensor(x), to_tensor(y)

        # if y shape is bigger, downscale it to x shape
        if self.resize_y and x.size()[-1] < y.size()[-1]:
            rs = Resize(size=x.size()[-1])
            y = rs(y)

        return {'x': x, 'y': y}
