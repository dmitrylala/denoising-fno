from pathlib import Path

import pandas as pd

__all__ = [
    'CSVSampler',
]


class CSVSampler:
    def __init__(self, fname: str | Path) -> None:
        self._head = -1
        self.source = pd.read_csv(fname)

    def __len__(self) -> int:
        return len(self.source)

    def __iter__(self) -> 'CSVSampler':
        return self

    def __call__(self) -> 'CSVSampler':
        return self.__iter__()

    def __next__(self) -> list[str]:
        # Reset sampler, if head is on the limits.
        if (self._head < 0) or (self._head >= self.__len__() - 1):
            self._head = -1
        # Locate head on the current element.
        self._head += 1
        # Return the current element.
        return self[self._head]

    def __getitem__(self, idx: int | slice) -> list[str]:
        return self.source.iloc[idx].tolist()
