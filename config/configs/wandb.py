from pydantic import BaseModel

__all__ = [
    'WandbConfig',
]


class WandbConfig(BaseModel):
    project: str = 'Denoising MRI'
    name: str
    group: str = 'FNO 2025'
    entity: str = 'Dmitrylala'
    tags: list[str]
