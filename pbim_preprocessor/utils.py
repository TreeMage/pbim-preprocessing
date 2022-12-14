from datetime import datetime
from typing import Optional, TypeVar, Generator, Any


class Logger:
    @staticmethod
    def log(level: str, msg: str, identifier: Optional[str] = None) -> None:
        print(
            f"[{datetime.now()}][{level}]{f'[{identifier}]' if identifier else ''} {msg}"
        )

    def info(self, msg: str, identifier: Optional[str] = None) -> None:
        self.log("INFO", msg, identifier)

    def error(self, msg: str, identifier: Optional[str] = None) -> None:
        self.log("ERROR", msg, identifier)

    def warn(self, msg: str, identifier: Optional[str] = None) -> None:
        self.log("WARN", msg, identifier)


LOGGER = Logger()


T = TypeVar("T")
U = TypeVar("U")


class GeneratorWithReturnValue:
    def __init__(self, gen: Generator[T, Any, U]):
        self.gen = gen

    def __iter__(self):
        self.value = yield from self.gen
