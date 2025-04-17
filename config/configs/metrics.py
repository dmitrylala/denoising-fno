from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict

__all__ = [
    'MetricConfig',
]


class MetricConfig(BaseModel):
    model_config = SettingsConfigDict(arbitrary_types_allowed=True)

    name: str
    cls: type
    kwargs: dict
