from pathlib import Path

from config import get_model_configs
from denoising import ModelRegistry


def test_model_registry_load() -> None:
    model_registry = ModelRegistry()
    model_registry.load(get_model_configs(Path('./notebooks')))
