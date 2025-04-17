from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from skimage.util import random_noise
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Transform
from tqdm import tqdm

from .data import BSD300Synthetic

__all__ = [
    'generate_synthetic',
    'load_bsd_synthetic',
]


def generate_synthetic(
    dset: Sequence[np.ndarray],
    variance: float,
    random_seed: int = 42,
    verbose: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    clean, noisy = [], []
    for img in tqdm(dset, disable=not verbose):
        img_f = img.astype(float) / 255.0
        img_noisy = random_noise(img_f, rng=random_seed, var=variance)
        clean.append(img_f)
        noisy.append(img_noisy)
    return clean, noisy


def load_bsd_synthetic(  # noqa: PLR0913
    root: str | Path,
    train_batch_size: int,
    test_batch_size: int,
    variance: float,
    transforms: Transform | None = None,
    device: torch.device = 'cpu',
) -> tuple[DataLoader, DataLoader]:
    dset_train = BSD300Synthetic(
        root,
        variance=variance,
        mode='train',
        device=device,
        transforms=transforms,
    )
    dset_test = BSD300Synthetic(root, variance=variance, mode='test', device=device)

    train_loader = DataLoader(dset_train, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(dset_test, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader
