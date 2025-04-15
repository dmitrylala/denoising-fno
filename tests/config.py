import os
from pathlib import Path

from config import get_datasets_configs, get_model_configs
from denoising import DatasetRegistry, ModelRegistry


def test_model_registry_load() -> None:
    model_registry = ModelRegistry()
    model_registry.load(get_model_configs(Path('./notebooks')))


def test_dataset_registry_creation() -> None:
    data_registry = DatasetRegistry()
    data_registry.load(get_datasets_configs(Path(os.getenv('HOME')) / 'data'))
