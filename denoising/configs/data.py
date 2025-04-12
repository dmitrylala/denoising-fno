from pydantic import BaseModel

from .common import ExistingPath

__all__ = [
    'BSDDatasetConfig',
    'FNODatasetConfig',
    'LoadParams',
]


class LoadParams(BaseModel):
    key: str
    shape: list[int]
    dtype: str


class BSDDatasetConfig(BaseModel):
    root: ExistingPath
    variance: float
    mode: str
    transforms: list | None = None


class FNODatasetConfig(BaseModel):
    root: ExistingPath
    sample_list: ExistingPath
    load_params: list[LoadParams]
    transforms: list | None = None
    normalize: bool = False
