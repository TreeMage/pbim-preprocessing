import datetime
import struct
from pathlib import Path
from typing import BinaryIO, Tuple, List

import numpy as np

from model.measurement import MeasurementFile, Measurement


MAGIC_FREQUENCY_CONSTANT = 270135


class PBimParser:
    def __init__(self):
        pass

    def parse(self, file_path: Path) -> MeasurementFile:
        name = file_path.name
        with open(file_path, "rb") as f:
            raw_timestamp, mittelungszahl = self._parse_header(f)
            measurements = self._parse_body(f, mittelungszahl)
        timestamp = (
            datetime.datetime.fromtimestamp(raw_timestamp)
            if raw_timestamp > 0
            else None
        )
        return MeasurementFile(
            name=name, description="", timestamp=timestamp, measurements=measurements
        )

    @staticmethod
    def _parse_header(f: BinaryIO) -> Tuple[int, int]:
        timestamp, mittelungszahl = struct.unpack("<ii", f.read(8))
        return timestamp, mittelungszahl

    @staticmethod
    def _parse_body(f: BinaryIO, mittelungszahl: int) -> List[Measurement]:
        data = np.fromfile(f, dtype=np.int32)
        frequency = MAGIC_FREQUENCY_CONSTANT / mittelungszahl
        return [
            Measurement(time=index / frequency, measurement=measurement)
            for index, measurement in enumerate(data)
        ]
