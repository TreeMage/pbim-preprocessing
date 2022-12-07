from enum import Enum
from pathlib import Path
from typing import List, Dict

from pbim_preprocessor.model import ChannelHeader, Measurement
from .data_parser import PBimDataParser
from .metadata_parser import PBimMetadataParser

MAGIC_FREQUENCY_CONSTANT = 270135

ALL_EXTENSIONS = ["R32", "DAT", "TSX", "events"]


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

    def parse(self, directory: Path, name: str):
        ensure_files_exist(directory, name, ALL_EXTENSIONS)
        global_header, channel_header = self._metadata_parser.parse(directory, name)
        data = self._data_parser.parse_all(directory, name, channel_header)
        time_channels = {
            channel.name: measurements
            for channel, measurements in data.items()
            if channel.name in TIME_CHANNELS
        }
        for channel in data.keys():
            if channel.name in POST_PROCESSABLE_CHANNELS:
                self._post_process(channel, data[channel], time_channels)
        return data

    @staticmethod
    def _post_process(
        channel: ChannelHeader,
        data: List[Measurement],
        time_channels: Dict[str, List[Measurement]],
    ) -> None:
        time_channel = CHANNEL_TIME_MAP[channel.name]
        if not time_channel:
            return
        time_data = time_channels[time_channel.value]
        assert len(time_data) == len(data)
        for i, dat in enumerate(data):
            dat.time = time_data[i].measurement
