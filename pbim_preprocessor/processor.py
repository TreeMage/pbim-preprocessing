import abc
import os
import re
import shutil
import struct
import zipfile
from pathlib import Path
from typing import List, Dict

from pbim_preprocessor.model import ParsedPBimChannel, Measurement
from pbim_preprocessor.parser.pbim import PBimParser
from pbim_preprocessor.utils import LOGGER

FILE_NAME_PATTERN = re.compile(r"^Job1_(\d{4})_(\d{2})_(\d{2})_.*$")
MEASUREMENT_SIZE_IN_BYTES = 12


class Processor(abc.ABC):
    @abc.abstractmethod
    def process(self):
        pass


class PBIMProcessor(Processor):
    def __init__(
        self,
        zip_file_path: Path,
        output_base_path: Path,
        tmp_path: Path,
        file_names: List[str],
        channels: List[str],
    ):
        self._zip_file_path = zip_file_path
        self._output_base_path = output_base_path
        self._tmp_path = tmp_path
        self._names = file_names
        self._channels = channels
        self._parser = PBimParser()
        self._identifier = os.getpid()

    def process(self):
        total = len(self._names)
        for i, raw_name in enumerate(self._names):
            LOGGER.info(
                f"Started processing '{raw_name}' ({i + 1}/{total})", self._identifier
            )
            self._parser.reset()
            path = self._unpack(raw_name)
            name = Path(raw_name).stem
            LOGGER.info(f"Unpacked '{name}' ({i + 1}/{total})", self._identifier)
            data = self._parser.parse(path, name, self._channels)
            LOGGER.info(f"Parsed '{name}' ({i + 1}/{total})", self._identifier)
            self.serialize(data, name)
            LOGGER.info(f"Saved '{name}' ({i + 1}/{total})", self._identifier)
            shutil.rmtree(path)
            LOGGER.info(
                f"Finished processing '{name}' ({i + 1}/{total})", self._identifier
            )

    def _unpack(self, name: str) -> Path:
        tmp_path = self._tmp_path / name
        tmp_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self._zip_file_path, "r") as zip_file:
            for extension in ["DAT", "R32"]:
                data = zip_file.read(f"{name}.{extension}")
                out_file_path = tmp_path / f"{Path(name).stem}.{extension}"
                out_file_path.write_bytes(data)
        return tmp_path

    def serialize(self, data: Dict[str, ParsedPBimChannel], name: str):
        for channel_name, parsed_channel in data.items():
            path = self.get_output_path(name, channel_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            serialized = b"".join(
                [self._serialize_measurement(m) for m in parsed_channel.measurements]
            )
            with open(path, "a+b") as f:
                f.write(serialized)

    @staticmethod
    def _serialize_measurement(measurement: Measurement) -> bytes:
        return struct.pack("<qf", measurement.time, measurement.measurement)

    def get_output_path(self, file_name: str, channel_name: str) -> Path:
        match = FILE_NAME_PATTERN.match(file_name)
        if not match:
            raise ValueError(
                f"File name '{file_name}' does not match the expected pattern"
            )
        year, month, day = match.groups()
        return (
            self._output_base_path
            / f"{year}"
            / f"{month}"
            / f"{day}"
            / f"{channel_name}.dat"
        )
