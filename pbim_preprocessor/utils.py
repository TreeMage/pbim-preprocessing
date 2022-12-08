from datetime import datetime
from typing import Optional


class Logger:
    @staticmethod
    def log(level: str, msg: str, identifier: Optional[str] = None) -> None:
        print(f"[{datetime.now()}][{level}]{f'[{identifier}]' if identifier else ''} {msg}")

    def info(self, msg: str, identifier: Optional[str] = None) -> None:
        self.log("INFO", msg, identifier)

    def error(self, msg: str, identifier: Optional[str] = None) -> None:
        self.log("ERROR", msg, identifier)

    def warn(self, msg: str, identifier: Optional[str] = None) -> None:
        self.log("WARN", msg, identifier)


LOGGER = Logger()
