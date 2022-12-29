from datetime import datetime
from typing import Optional, TypeVar, Generator, Any


class Logger:
    def __init__(self, debug: bool = False) -> None:
        self._debug = debug

    def set_debug(self, debug: bool) -> None:
        self._debug = debug

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

    def debug(self, msg: str, identifier: Optional[str] = None) -> None:
        if self._debug:
            self.log("DEBUG", msg, identifier)


LOGGER = Logger()


T = TypeVar("T")
U = TypeVar("U")


class GeneratorWithReturnValue:
    def __init__(self, gen: Generator[T, Any, U]):
        self.gen = gen

    def __iter__(self):
        self.value = yield from self.gen
