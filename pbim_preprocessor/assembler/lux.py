import datetime
import struct
import zipfile
from pathlib import Path
from typing import Generator, Dict, Any, List, Optional, Tuple, BinaryIO

import numpy as np

from pbim_preprocessor.cli.merge import _load_index
from pbim_preprocessor.metadata import DatasetMetadata
from pbim_preprocessor.model import EOF, Measurement
from pbim_preprocessor.parser.lux import LuxAccelerationParser, LuxTemperatureParser
from pbim_preprocessor.sampling import SamplingStrategy
from pbim_preprocessor.utils import LOGGER, _load_metadata


class LuxAssembler:

    PREFIX_PATH = "DynamicMeasurements"
    TEMPERATURE_PATH = f"displacement and temperatures.xlsx"

    def __init__(self, file_path: Path, resolution: float, strategy: SamplingStrategy):
        self._acceleration_parser = LuxAccelerationParser()
        self._temperature_parser = LuxTemperatureParser()
        self._file_path = file_path
        self._resolution = resolution
        self._strategy = strategy

    def _find_folder_for_time(
        self, zip_file: zipfile.ZipFile, time: datetime.datetime
    ) -> Optional[str]:
        folder_name = time.strftime("%Y%m%d")
        prefixed_name = f"{self.PREFIX_PATH}/{folder_name}/"
        return folder_name if prefixed_name in zip_file.namelist() else None

    @staticmethod
    def _find_sub_folder_for_time(
        zip_file: zipfile.ZipFile, folder: str, time: datetime.datetime
    ) -> Optional[str]:
        sub_folder_name = time.strftime("%d%m%Y_%H%M%S")
        return (
            sub_folder_name
            if f"{LuxAssembler.PREFIX_PATH}/{folder}/{sub_folder_name}/"
            in zip_file.namelist()
            else None
        )

    def _advance_time_until_folder_exists(
        self,
        zip_file: zipfile.ZipFile,
        time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> Optional[datetime.datetime]:
        def _folder_date(name: str) -> datetime.datetime:
            return datetime.datetime.strptime(Path(name).name, "%d%m%Y_%H%M%S").replace(
                tzinfo=datetime.timezone.utc
            )

        while self._find_folder_for_time(zip_file, time) is None and time < end_time:
            time += datetime.timedelta(days=1)
        if time > end_time:
            return None
        year, month, day = time.year, time.month, time.day
        LOGGER.debug(f"Searching folder for time {year}-{month:02d}-{day:02d}")
        folder = self._find_folder_for_time(
            zip_file, datetime.datetime(year, month, day)
        )
        LOGGER.debug(f"Found folder '{folder}'. Searching sub folders...")
        sub_folders = sorted(
            self._list_zip_folder_sub_folders(zip_file, folder),
            key=lambda name: _folder_date(Path(name).name),
        )
        LOGGER.debug(f"Found sub folders: {sub_folders}")
        for i, sub_folder in enumerate(sub_folders):
            folder_date = _folder_date(sub_folder)
            LOGGER.debug(
                f"Checking sub folder '{sub_folder}' with date {folder_date} for target time {time}"
            )
            if folder_date >= time:
                LOGGER.debug(f"Match.")
                return folder_date
            LOGGER.debug("No match.")
        LOGGER.debug(f"No match in current folder, advancing to next folder.")
        return self._advance_time_until_folder_exists(
            zip_file,
            datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)
            + datetime.timedelta(days=1),
            end_time,
        )

    def _parse_data(
        self, zip_file: zipfile.ZipFile, folder: str
    ) -> Tuple[List[str], List[np.ndarray]]:
        f = zip_file.open(f"{self.PREFIX_PATH}/{folder}/Acceleration.tdms", "r")
        channels, data = self._acceleration_parser.parse(f)
        return channels, data

    def _sample_channels(self, data: np.ndarray, time: np.ndarray):
        # We assume all channels to have the same time stamps.
        end_time = time[-1]
        sampled_data = []
        current_idx = 0
        current_time = time[0]
        current_end_time = time[0] + self._resolution
        current_sample = []
        while current_time < end_time:
            while current_time < current_end_time and current_time < end_time:
                current_time = time[current_idx]
                current_sample.append(
                    Measurement(data[current_idx], time=int(current_time * 1000))
                )
                current_idx += 1

            sampled_data.append(
                self._strategy.sample(
                    current_sample,
                    datetime.datetime.fromtimestamp(
                        current_end_time - self._resolution / 2
                    ),
                )
            )
            current_end_time += self._resolution
            current_sample = []

        return sampled_data

    def _list_zip_folder_sub_folders(
        self, zip_file: zipfile.ZipFile, folder: str
    ) -> List[str]:
        path = f"{self.PREFIX_PATH}/{folder}/"
        return [
            name
            for name in zip_file.namelist()
            if name.startswith(path) and name != path and name.endswith("/")
        ]

    @staticmethod
    def _find_temperature_for_time(temperatures: np.ndarray, time: float) -> float:
        differences = np.abs(temperatures[:, 0] - time)
        return temperatures[np.argmin(differences)][1]

    def _assemble_from_zip(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        channels: List[str],
    ) -> Generator[Dict[str, float] | EOF, Any, None]:
        parse_temperature = "Temperature" in channels
        sanitized_channels = [
            channel for channel in channels if channel != "Temperature"
        ]
        with zipfile.ZipFile(self._file_path, "r") as zip_file:
            if parse_temperature:
                LOGGER.info("Parsing temperatures.")
                with zip_file.open(self.TEMPERATURE_PATH, "r") as f:
                    temperatures = self._temperature_parser.parse(f)
            LOGGER.info(f"Searching folder for time {start_time}.")
            current_time = self._advance_time_until_folder_exists(
                zip_file, start_time, end_time
            )
            if current_time is None:
                raise ValueError(f"Could not find folder for time {start_time}.")
            while current_time < end_time:
                current_folder = self._find_folder_for_time(zip_file, current_time)
                sub_folder = self._find_sub_folder_for_time(
                    zip_file, current_folder, current_time
                )
                LOGGER.info(f"Processing folder {sub_folder} for time {current_time}.")
                LOGGER.info("Parsing data.")
                parsed_channels, data = self._parse_data(
                    zip_file, f"{current_folder}/{sub_folder}"
                )
                time = data[parsed_channels.index("time")]
                LOGGER.info(f"Sampling data for time {current_time}.")
                if self._resolution > 0:
                    sampled_data = [
                        self._sample_channels(
                            data[parsed_channels.index(channel)], time
                        )
                        for channel in sanitized_channels
                    ]
                else:
                    sampled_data = [
                        data[parsed_channels.index(channel)]
                        for channel in sanitized_channels
                    ]
                sampled_time = (
                    np.arange(time[0], time[-1], self._resolution)
                    if self._resolution > 0
                    else time
                )
                for step in range(len(sampled_data[0])):
                    LOGGER.info(
                        f"Yielding data for time {current_time} at step {step+1}/{len(sampled_data[0])}."
                    )
                    step_data = {
                        channel: sampled_data[i][step]
                        for i, channel in enumerate(sanitized_channels)
                    }
                    adjusted_time = current_time + datetime.timedelta(
                        seconds=sampled_time[step]
                    )
                    timestamp_in_ms = adjusted_time.timestamp() * 1000
                    step_data["time"] = int(timestamp_in_ms)
                    if parse_temperature:
                        step_data["Temperature"] = self._find_temperature_for_time(
                            temperatures, timestamp_in_ms
                        )
                    yield step_data
                LOGGER.info(
                    f"Finished processing folder {sub_folder} for time {current_time}."
                )
                yield EOF()
                current_time = self._advance_time_until_folder_exists(
                    zip_file,
                    current_time + datetime.timedelta(seconds=time[-1]),
                    end_time,
                )
                if current_time is None:
                    LOGGER.warn(
                        f"Reached end time while searching for next folder. Stopping."
                    )
                    break

    @staticmethod
    def _load_pre_assembled(
        f: BinaryIO, index: int, metadata: DatasetMetadata
    ) -> Dict[str, float]:
        f.seek(index * metadata.measurement_size_in_bytes)
        data = f.read(metadata.measurement_size_in_bytes)
        format_str = f"<{'q' if metadata.time_byte_size == 8 else 'i'}{'f' * (len(metadata.channel_order) - 1)}"
        parsed = struct.unpack(format_str, data)
        return {
            channel: value for channel, value in zip(metadata.channel_order, parsed)
        }

    @staticmethod
    def _make_datetime(time: float) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(time / 1000, tz=datetime.timezone.utc)

    def _assemble_from_pre_assembled(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        channels: List[str],
    ) -> Generator[Dict[str, float] | EOF, Any, None]:
        metadata = _load_metadata(self._file_path)
        index = _load_index(self._file_path)
        f = open(self._file_path, "rb")
        target_time = start_time + datetime.timedelta(seconds=self._resolution)
        samples = []
        for idx, entry in enumerate(index.entries):
            sample = self._load_pre_assembled(f, entry.end_measurement_index, metadata)
            if self._make_datetime(sample["Time"]) < start_time:
                LOGGER.info(f"Skipping index entry {idx + 1} since it is too early.")
                continue
            for i in range(entry.start_measurement_index, entry.end_measurement_index):
                sample = self._load_pre_assembled(f, i, metadata)
                time = self._make_datetime(sample["Time"])
                if time < start_time:
                    continue
                if time > end_time or target_time > end_time:
                    return
                if self._resolution > 0:
                    if time <= target_time:
                        samples.append(
                            {
                                channel: Measurement(
                                    measurement=sample[channel],
                                    time=int(time.timestamp() * 1000),
                                )
                                for channel in channels
                            }
                        )
                    else:
                        sampled = {
                            channel: self._strategy.sample(
                                [s[channel] for s in samples],
                                target_time
                                - datetime.timedelta(seconds=self._resolution / 2),
                            )
                            for channel in channels
                        }
                        yield sampled
                        samples = []
                        target_time += datetime.timedelta(seconds=self._resolution)
                else:
                    yield {channel: sample[channel] for channel in channels}
            yield EOF()

    def assemble(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        channels: List[str],
    ) -> Generator[Dict[str, float] | EOF, Any, None]:
        if self._file_path.name.endswith(".zip"):
            LOGGER.info(f"Using zip file {self._file_path}.")
            yield from self._assemble_from_zip(start_time, end_time, channels)
        elif self._file_path.name.endswith(".dat"):
            LOGGER.info(f"Using pre-assembled data from {self._file_path}.")
            yield from self._assemble_from_pre_assembled(start_time, end_time, channels)
        else:
            raise ValueError(f"Unknown file type {self._file_path}.")
