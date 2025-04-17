from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict

from .common import ExistingPath

__all__ = [
    'ModelConfig',
]


class ModelConfig(BaseModel):
    model_config = SettingsConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    fno_cfg: dict
    weights_path: ExistingPath | None
    model_class_name: str
