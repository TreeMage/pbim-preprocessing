import abc
from pathlib import Path
from typing import List, Dict


class Writer(abc.ABC):

    @abc.abstractmethod
    def __enter__(self):
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abc.abstractmethod
    def write_step(self, data: Dict[str, float]) -> None:
        pass


class CsvWriter(Writer):
    def __init__(self, path: Path, header: List[str], delimiter=","):
        self._path = path
        self._header = header
        self._delimiter = delimiter
        self._f = None

    def __enter__(self):
        self._f = open(self._path, "w")
        self._f.write(self._delimiter.join(self._header) + "\n")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._f.close()

    def write_step(self, data: Dict[str, float]) -> None:
        line = self._delimiter.join([f"{data[header]:.6f}" for header in self._header])
        self._f.write(line + "\n")

