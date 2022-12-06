from dataclasses import dataclass
from typing import Optional


@dataclass
class Measurement:
    measurement: float
    time: Optional[float] = None
