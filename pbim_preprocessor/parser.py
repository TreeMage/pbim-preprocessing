import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict

from pbim_preprocessor.model import Measurement, ParsedChannel
from .data_parser import PBimDataParser
from .metadata_parser import PBimMetadataParser

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
    ) -> Dict[str, ParsedChannel]:
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
        channel: ParsedChannel,
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
