from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.io import loadmat

__all__ = [
    'Batch',
    'collate_wrapper',
    'load_grayscale',
    'load_txt',
    'read_img',
]


class Batch:
    def __init__(self, data: list[dict[str, torch.Tensor]]) -> None:
        self.x = torch.stack([s['x'] for s in data], 0).cpu()
        self.y = torch.stack([s['y'] for s in data], 0).cpu()

    def pin_memory(self) -> 'Batch':
        self.x = self.x.pin_memory()
        self.y = self.y.pin_memory()
        return self

    def items(self):  # noqa: ANN201
        return {'x': self.x, 'y': self.y}.items()


def collate_wrapper(batch: list[dict[str, torch.Tensor]]):  # noqa: ANN201
    return Batch(batch)


def _as_hwc(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:  # noqa: PLR2004
        img = img[..., np.newaxis]
    elif img.ndim != 3:  # noqa: PLR2004
        msg = f'Expected 2 or 3 dimensional image, found ndim: {img.ndim}.'
        raise ValueError(msg)
    return img


def decode_raw(read_path: str | Path, dtype: str, shape: list[int]) -> np.ndarray:
    img = np.fromfile(read_path, dtype=dtype)
    return np.reshape(img, shape, order='C')


def decode_png(read_path: str | Path) -> np.ndarray:
    img = cv2.imread(read_path, -1)
    img = _as_hwc(img)
    if img.shape[-1] == 3:  # noqa: PLR2004
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.shape[-1] == 4:  # noqa: PLR2004
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img


def decode_mat(read_path: str | Path, key: str) -> np.ndarray:
    return np.asarray(loadmat(read_path)[key])


def read_img(
    read_path: str | Path,
    dtype: str | None = None,
    shape: list[int] | None = None,
    key: str | None = None,
    normalize: float | None = None,
) -> np.ndarray:
    # Setup.
    read_path = Path(read_path)
    if not read_path.exists():
        msg = f'Source path does not exist: {read_path}.'
        raise ValueError(msg)
    ext = read_path.suffix.lower()

    # Read.
    if ext in ('.png', '.jpeg', '.jpg'):
        img = decode_png(read_path)
    elif ext in ('.raw', '.bin'):
        img = decode_raw(read_path, dtype, shape)
    elif ext in ('.mat',):
        img = decode_mat(read_path, key)
    else:
        msg = f'Unrecognized filename extension found: {read_path}. Only `.png`, `.jpeg`, `.jpg`, `.raw`, `.bin` or `.mat` are supported.'  # noqa: E501
        raise ValueError(msg)

    # Process.
    if dtype is not None:
        assert img.dtype == np.dtype(dtype), f'`dtype` mismatch: `{img.dtype}` != `{dtype}`.'  # noqa: S101
    if shape is not None and all(shape):
        assert img.shape == tuple(shape), f'`shape` mismatch: `{img.shape}` != `{shape}`.'  # noqa: S101

    if normalize is not None:
        img = img / float(normalize)

    return img


def load_grayscale(path: str) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def load_txt(path: str) -> list[str]:
    with Path.open(path) as f:
        return list(map(str.strip, f.readlines()))
