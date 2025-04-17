import random

import cv2
import numpy as np
import torch

__all__ = [
    'save_grayscale',
    'seed_everything',
]


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)


def save_grayscale(path: str, image: np.ndarray) -> None:
    cv2.imwrite(str(path), image)
