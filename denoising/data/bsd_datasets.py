from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms.v2 import Transform

from .utils import load_grayscale, load_txt

__all__ = [
    'BSD300',
    'BSD300Synthetic',
]


class BSD300:
    def __init__(self, root: str, mode: str = 'train') -> None:
        if mode not in ('train', 'test'):
            msg = f"mode should be one of 'train', 'test', got: {mode}"
            raise ValueError(msg)

        root = Path(root)
        image_ids = load_txt(root / f'iids_{mode}.txt')
        self.images = [load_grayscale(root / f'images/{mode}/{img_id}.jpg') for img_id in image_ids]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int | slice) -> np.ndarray:
        return self.images[idx]


class BSD300Synthetic(Dataset):
    def __init__(
        self,
        root: str,
        variance: float,
        mode: str = 'train',
        transforms: Transform | None = None,
    ) -> None:
        if mode not in ('train', 'test'):
            msg = f"mode should be one of 'train', 'test', got: {mode}"
            raise ValueError(msg)

        folder = Path(root) / f'gauss-noise-{variance}' / mode
        if not folder.exists():
            msg = f'Folder {folder} not exists'
            raise ValueError(msg)

        noisy_paths = sorted(folder.glob('clean*.jpg'))
        clean_paths = sorted(folder.glob('noisy*.jpg'))
        self.paths = list(zip(noisy_paths, clean_paths, strict=True))
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int | slice) -> dict[str, torch.Tensor]:
        noisy_path, clean_path = self.paths[index]

        def to_tensor(img: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(img).unsqueeze(0).clone().type(torch.float32)

        x = to_tensor(load_grayscale(noisy_path))
        y = to_tensor(load_grayscale(clean_path))

        if self.transforms is not None:
            inp = torch.cat([x, y], axis=0)
            inp_transformed = self.transforms(inp)
            x = inp_transformed[0:1, :, :]
            y = inp_transformed[1:, :, :]

        return {'x': x.clone(), 'y': y.clone()}
