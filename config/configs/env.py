from typing import Annotated

from pydantic import AfterValidator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .common import ExistingPath

WANDB_KEY_LENGTH = 40

__all__ = [
    'Environment',
]


def check_key_length(key: str) -> str:
    if len(key) != WANDB_KEY_LENGTH:
        msg = f'Wandb key should has {WANDB_KEY_LENGTH} symbols, got key: {key}'
        raise ValueError(msg)
    return key


WandApiKey = Annotated[str, AfterValidator(check_key_length)]


class Environment(BaseSettings):
    model_config = SettingsConfigDict(env_file='env', env_file_encoding='utf-8')

    data: ExistingPath
    weights: ExistingPath
    wandb_api_key: str
