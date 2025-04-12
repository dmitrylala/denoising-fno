from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator

__all__ = [
    'ExistingPath',
]


def path_exists(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        msg = f"Path {p!r} doesn't exist"
        raise ValueError(msg)
    return p


ExistingPath = Annotated[Path, AfterValidator(path_exists)]
