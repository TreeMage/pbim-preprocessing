from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class Measurement:
    time: float
    measurement: float


@dataclass
class MeasurementFile:
    name: str
    description: str
    timestamp: Optional[datetime]
    measurements: List[Measurement]

    def __str__(self):
        return f"MeasurementFile(name={self.name}, description={self.description}, timestamp={self.timestamp}, #measurements={len(self.measurements)}"
