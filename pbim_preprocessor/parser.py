import datetime
import os
import re
import tempfile
import zipfile
import io
from enum import Enum
from pathlib import Path
from typing import List, Dict, Generator, Any, Literal

from pbim_preprocessor.model import Measurement, ParsedPBimChannel, ParsedZ24File, EOF
from pbim_preprocessor.data_parser import (
    PBimDataParser,
    Z24AccelerationDataParser,
    Z24EnvironmentalDataParser,
    Z24PDTAccelerationParser,
)
from pbim_preprocessor.metadata_parser import PBimMetadataParser
from pbim_preprocessor.utils import LOGGER

MAGIC_FREQUENCY_CONSTANT = 270135

ALL_EXTENSIONS = ["R32", "DAT"]


def ensure_files_exist(directory: Path, name: str, extensions: List[str]) -> None:
    for extension in extensions:
        if not (directory / f"{name}.{extension}").exists():
            raise MeasurementFileMissingError(name, extension)


class MeasurementFileMissingError(Exception):
    def __init__(self, name: str, extension: str):
        self._name = name
        self._extensions = extension


class TimeChannel(Enum):
    STANDARD_1 = "Zeit  1 - Standardmessrate"
    STANDARD_2 = "Zeit  2 - Standardmessrate"
    STANDARD_4 = "Zeit  4 - Standardmessrate"
    FAST_3 = "Zeit  3 - Schnelle Messrate"
    FAST_5 = "Zeit  5 - Schnelle Messrate"
    SLOW_5 = "Zeit  5 - Langsame Messrate"


CHANNEL_TIME_MAP = {
    # Measurement amplifier 1
    "MS1L": TimeChannel.STANDARD_1,
    "MS2L": TimeChannel.STANDARD_1,
    "MS3Q": TimeChannel.STANDARD_1,
    "MS3L": TimeChannel.STANDARD_1,
    "MS4Q": TimeChannel.STANDARD_1,
    "MS4L": TimeChannel.STANDARD_1,
    "MS5L": TimeChannel.STANDARD_1,
    "MS6L": TimeChannel.STANDARD_1,
    # Measurement amplifier 2
    "MS7Q": TimeChannel.STANDARD_2,
    "MS7L": TimeChannel.STANDARD_2,
    "MS8Q": TimeChannel.STANDARD_2,
    "MS8L": TimeChannel.STANDARD_2,
    "MS27v": TimeChannel.STANDARD_2,
    "MS28v": TimeChannel.STANDARD_2,
    "MS29v": TimeChannel.STANDARD_2,
    "MS30h": TimeChannel.STANDARD_2,
    # Measurement amplifier 3
    "MS23": TimeChannel.FAST_3,
    "MS24": TimeChannel.FAST_3,
    "MS25": TimeChannel.FAST_3,
    "MS26": TimeChannel.FAST_3,
    # Measurement amplifier 4
    "MS9Q": TimeChannel.STANDARD_4,
    "MS10Q": TimeChannel.STANDARD_4,
    "MS11Q": TimeChannel.STANDARD_4,
    "MS12Q": TimeChannel.STANDARD_4,
    "MS13Q": TimeChannel.STANDARD_4,
    "MS14Q": TimeChannel.STANDARD_4,
    "MS15Q": TimeChannel.STANDARD_4,
    "MS16Q": TimeChannel.STANDARD_4,
    # Measurement amplifier 5
    "MS17": TimeChannel.FAST_5,
    "MS18": TimeChannel.FAST_5,
    "MS19": TimeChannel.FAST_5,
    "MS20": TimeChannel.FAST_5,
    "MS21": TimeChannel.FAST_5,
    "MS22": TimeChannel.FAST_5,
    "MS6T1u": TimeChannel.SLOW_5,
    "MS6T2o": TimeChannel.SLOW_5,
}

POST_PROCESSABLE_CHANNELS = list(CHANNEL_TIME_MAP.keys())
TIME_CHANNELS = [channel.value for channel in TimeChannel]


class PBimParser:
    def __init__(self):
        self._metadata_parser = PBimMetadataParser()
        self._data_parser = PBimDataParser()

    def parse(
        self, directory: Path, name: str, channels: List[str]
    ) -> Dict[str, ParsedPBimChannel]:
        ensure_files_exist(directory, name, ALL_EXTENSIONS)
        channels_with_time = channels + [channel.value for channel in TimeChannel]
        global_header, channel_header = self._metadata_parser.parse(directory, name)
        start_date, start_time = global_header.date, global_header.time
        t0 = datetime.datetime(
            year=start_date.year,
            month=start_date.month,
            day=start_date.day,
            hour=start_time.hour,
            minute=start_time.minute,
            second=start_time.second,
            microsecond=start_time.microsecond,
        )
        filtered_headers = [
            header for header in channel_header if header.name in channels_with_time
        ]
        data = self._data_parser.parse_all(directory, name, filtered_headers)
        time_channels = {
            name: parsed_channel.measurements
            for name, parsed_channel in data.items()
            if name in TIME_CHANNELS
        }
        for name, parsed_channel in data.items():
            if name in POST_PROCESSABLE_CHANNELS and name in channels:
                self._post_process(parsed_channel, time_channels, t0)
        return {
            name: parsed_channel
            for name, parsed_channel in data.items()
            if name in channels
        }

    def reset(self):
        self._metadata_parser.reset()

    @staticmethod
    def _post_process(
        channel: ParsedPBimChannel,
        time_channels: Dict[str, List[Measurement]],
        start_time: datetime.datetime,
    ) -> None:
        time_channel = CHANNEL_TIME_MAP[channel.channel_header.name]
        if not time_channel:
            return
        time_data = time_channels[time_channel.value]
        assert len(time_data) == len(channel.measurements)
        start_time_in_milliseconds = int(start_time.timestamp() * 1000)
        for i, dat in enumerate(channel.measurements):
            # Round to the nearest millisecond
            dat.time = start_time_in_milliseconds + int(time_data[i].measurement * 1000)


Z24_EMS_REGEX = re.compile(r"Z24ems(\d+).zip")
Z24_INNER_ZIP_REGEX = re.compile(r"(\d{2})([A-G])(\d{2}).zip")


class Z24UndamagedParser:
    def __init__(self):
        self._acceleration_parser = Z24AccelerationDataParser()
        self._environmental_data_parser = Z24EnvironmentalDataParser()

    @staticmethod
    def _find_and_sort_ems_files(directory: Path) -> List[Path]:
        files = [f for f in os.listdir(directory) if Z24_EMS_REGEX.match(f)]
        files = sorted(files, key=lambda f: int(Z24_EMS_REGEX.match(f).group(1)))
        return [directory / f for f in files]

    @staticmethod
    def _parser_inner_file_name(name: str) -> datetime.datetime:
        match = Z24_INNER_ZIP_REGEX.match(name)
        if not match:
            raise ValueError(f"Could not parse {name}")
        week = int(match.group(1)) - 1  # 0-indexed
        day = ord(match.group(2)) - ord("A") - 2  # 0-indexed but C is Monday
        hour = int(match.group(3))
        return datetime.datetime(
            year=1997, month=11, day=10, tzinfo=datetime.timezone.utc
        ) + datetime.timedelta(weeks=week, days=day, hours=hour)

    @staticmethod
    def _extract_inner_file(
        zip_file: zipfile.ZipFile, inner_file: str, output_path: Path
    ) -> Path:
        zip_file.extract(inner_file, path=output_path)
        return output_path / inner_file

    def _parse_inner_file(self, inner_file_path: Path) -> ParsedZ24File:
        with zipfile.ZipFile(inner_file_path) as inner_zip_file:
            data = {}
            pre_env_measurements = None
            post_env_measurements = None
            files = inner_zip_file.namelist()
            for file in files:
                f = io.TextIOWrapper(
                    inner_zip_file.open(file, "r"), encoding="ISO-8859-1", newline=None
                )
                if file.endswith(".aaa"):
                    if "car" in file:
                        continue
                    channel = file[-6:-4]
                    data[channel] = self._acceleration_parser.parse(f)
                elif file.endswith(".env"):
                    if "PRE" in file:
                        pre_env_measurements = self._environmental_data_parser.parse(f)
                    elif "POS" in file:
                        post_env_measurements = self._environmental_data_parser.parse(f)
                    else:
                        raise ValueError(f"Unknown env file {file}")
                else:
                    raise ValueError(f"Unknown file type {file}")
            if not pre_env_measurements or not post_env_measurements:
                raise ValueError(f"Missing env file in {inner_file_path}")
            return ParsedZ24File(
                pre_measurement_environmental_data=pre_env_measurements,
                post_measurement_environmental_data=post_env_measurements,
                acceleration_data=data,
            )

    def _parse_ems_file(
        self,
        ems_file_path: Path,
        output_path: Path,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> Generator[ParsedZ24File, Any, None]:
        with zipfile.ZipFile(ems_file_path) as ems_file:
            inner_files = sorted(ems_file.namelist(), key=self._parser_inner_file_name)
            for inner_file in inner_files:
                LOGGER.info(f"Parsing {inner_file}")

                inner_file_path = self._extract_inner_file(
                    ems_file, inner_file, output_path
                )
                time = self._parser_inner_file_name(inner_file)
                if time < start_time:
                    continue
                if time > end_time:
                    break
                yield self._parse_inner_file(inner_file_path)
                inner_file_path.unlink()

    def parse(
        self,
        path: Path,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> Generator[ParsedZ24File, Any, None]:
        LOGGER.info(f"Parsing Z24 undamaged data from {path} between {start_time} and {end_time}")
        if path.is_dir():
            LOGGER.info("Given path is a directory, searching for EMS files")
            ems_files = self._find_and_sort_ems_files(path)
            LOGGER.info(f"Found {len(ems_files)} EMS files")
        else:
            ems_files = [path]
        with tempfile.TemporaryDirectory() as temp_dir:
            for ems_file_path in ems_files:
                LOGGER.info(f"Parsing {ems_file_path}")
                yield from self._parse_ems_file(
                    ems_file_path, Path(temp_dir), start_time, end_time
                )


class Z24DamagedParser:
    NUM_SCENARIOS = 9

    def __init__(self):
        self._data_parser = Z24PDTAccelerationParser()

    @staticmethod
    def _make_data_path(scenario: int, mode: Literal["avt", "fvt"], index: int) -> str:
        return f"{scenario:02d}/{mode}/{scenario:02d}setup{index:02d}.mat"

    def parse(
        self, zip_path: Path, scenario: int, mode: Literal["avt", "fvt"]
    ) -> ParsedZ24File:
        LOGGER.info(f"Parsing Z24 damaged data from {zip_path}")
        with zipfile.ZipFile(zip_path) as zip_file:
            in_file_paths = [
                self._make_data_path(scenario, mode, i)
                for i in range(1, self.NUM_SCENARIOS + 1)
            ]
            with tempfile.TemporaryDirectory() as temp_dir:
                paths = [
                    Path(zip_file.extract(path, path=temp_dir))
                    for path in in_file_paths
                ]
                data = [self._data_parser.parse(path) for path in paths]
        merged = {}
        for scenario in data:
            merged.update(scenario)
        return ParsedZ24File(
            pre_measurement_environmental_data={},
            post_measurement_environmental_data={},
            acceleration_data=merged,
        )
