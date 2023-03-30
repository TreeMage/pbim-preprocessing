import abc
import calendar
import datetime
import struct
from pathlib import Path
from typing import List, BinaryIO, Optional, Tuple

import numpy as np
import tqdm

from pbim_preprocessor.cli.merge import _load_index
from pbim_preprocessor.index import CutIndexEntry, _write_index, _write_index_file
from pbim_preprocessor.metadata import DatasetMetadata, _write_metadata_file
from pbim_preprocessor.utils import _load_metadata


class DatasetSamplingStrategy(abc.ABC):
    @abc.abstractmethod
    def compute_sample_indices(self, time: np.ndarray) -> List[int]:
        pass


class UniformSamplingStrategy(DatasetSamplingStrategy):
    def __init__(self, num_samples: int):
        self._num_samples = num_samples

    def compute_sample_indices(self, time: np.ndarray) -> List[int]:
        return np.linspace(0, len(time) - 1, self._num_samples, dtype=np.int).tolist()


class RandomSamplingStrategy(DatasetSamplingStrategy):
    def __init__(self, num_samples: int):
        self._num_samples = num_samples

    def compute_sample_indices(self, time: np.ndarray) -> List[int]:
        return np.random.choice(len(time), self._num_samples, replace=False).tolist()


class MinutesPerHourSamplingStrategy(DatasetSamplingStrategy):
    def __init__(self, number_of_minutes_per_hour: int):
        self._number_of_minutes_per_hour = number_of_minutes_per_hour

    @staticmethod
    def _make_datetime(timestamp: float) -> datetime.datetime:
        return datetime.datetime.utcfromtimestamp(timestamp / 1000)

    @staticmethod
    def _make_timestamp(dt: datetime.datetime) -> float:
        return calendar.timegm(dt.timetuple()) * 1000

    def compute_sample_indices(self, time: np.ndarray) -> List[int]:
        final_indices = []
        start_date = self._make_datetime(time[0])
        end_date = self._make_datetime(time[-1])

        current_date = start_date
        while current_date < end_date:
            next_date = current_date + datetime.timedelta(hours=1)
            indices = np.where(
                (time >= self._make_timestamp(current_date))
                & (time < self._make_timestamp(next_date))
            )[0]
            spacing = len(indices) // self._number_of_minutes_per_hour
            for i in range(self._number_of_minutes_per_hour):
                offset = i * spacing
                end = offset + 60 * 75
                if end > len(indices):
                    end = len(indices)
                final_indices.extend(indices[offset:end])
            current_date = next_date
        return final_indices


class PBimSampler:
    def __init__(
        self,
        window_size: int,
        remove_zero_windows: bool,
        sampling_strategy: DatasetSamplingStrategy,
    ):
        self._window_size = window_size
        self._remove_zero_windows = remove_zero_windows
        self._sampling_strategy = sampling_strategy

    @staticmethod
    def _load_raw_samples(
        f: BinaryIO,
        metadata: DatasetMetadata,
        start_index: int,
        end_index: Optional[int],
    ) -> bytes:
        f.seek(start_index * metadata.measurement_size_in_bytes)
        num_measurements = end_index - start_index if end_index is not None else 1
        return f.read(num_measurements * metadata.measurement_size_in_bytes)

    def _load_window(self, f: BinaryIO, index: int, metadata: DatasetMetadata):
        assert index <= metadata.length - self._window_size + 1
        f.seek(index * metadata.measurement_size_in_bytes)
        buffer = self._load_raw_samples(f, metadata, index, index + self._window_size)

        data = np.zeros((len(metadata.channel_order), self._window_size))
        time_byte_format = "<q" if metadata.time_byte_size == 8 else "<i"
        format_string = time_byte_format + "f" * (len(metadata.channel_order) - 1)
        for i in range(self._window_size):
            offset = i * metadata.measurement_size_in_bytes
            measurement = np.array(
                struct.unpack(
                    format_string,
                    buffer[offset : offset + metadata.measurement_size_in_bytes],
                )
            )
            data[:, i] = measurement
        return data

    @staticmethod
    def _load_time(f: BinaryIO, metadata: DatasetMetadata) -> np.ndarray:
        f.seek(0)
        buffer = f.read()
        time_byte_format = "<q" if metadata.time_byte_size == 8 else "<i"
        return np.array(
            [
                struct.unpack(time_byte_format, buffer[i : i + metadata.time_byte_size])
                for i in range(0, len(buffer), metadata.measurement_size_in_bytes)
            ]
        ).reshape(-1)

    @staticmethod
    def _serialize_window(window: np.ndarray, metadata: DatasetMetadata) -> bytes:
        time_byte_format = "<q" if metadata.time_byte_size == 8 else "<i"
        format_string = time_byte_format + "f" * (len(metadata.channel_order) - 1)
        return b"".join(
            [struct.pack(format_string, *measurement) for measurement in window.T]
        )

    @staticmethod
    def _compute_start_and_end_indices(indices: List[int]) -> List[Tuple[int, int]]:
        computed_indices = []
        current_start_index = 0
        while current_start_index < len(indices):
            current_end_index = current_start_index + 1
            while (
                current_end_index < len(indices) - 1
                and indices[current_end_index] == indices[current_end_index + 1] - 1
            ):
                current_end_index += 1
            computed_indices.append(
                (current_start_index, int(min(current_end_index + 1, len(indices))))
            )
            current_start_index = current_end_index + 1
        return computed_indices

    @staticmethod
    def _is_anomalous(index: int, cut_index: List[CutIndexEntry]) -> bool:
        for entry in cut_index:
            if entry.start_measurement_index <= index < entry.end_measurement_index:
                return entry.anomalous
        raise ValueError(f"Index {index} is not in the cut index")

    def process(self, input_path: Path, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = _load_metadata(input_path)
        index = _load_index(input_path)
        number_of_windows = metadata.length - self._window_size + 1
        # number_of_windows = 100000
        indices = []
        with open(input_path, "rb") as f:
            for i in tqdm.trange(number_of_windows, desc="Loading windows"):
                window = self._load_window(f, i, metadata)
                window_without_time = window[1:, :]
                if self._remove_zero_windows and np.all(window_without_time == 0):
                    continue
                indices.append(i)
            time = self._load_time(f, metadata)
        time = time[indices]
        sample_indices = self._sampling_strategy.compute_sample_indices(time)
        index_entries = []
        with open(output_path, "wb") as output_file_handle:
            with open(input_path, "rb") as input_file_handle:
                for start, end in tqdm.tqdm(
                    self._compute_start_and_end_indices(sample_indices)
                ):
                    index_entries.append(
                        CutIndexEntry(
                            start_measurement_index=start,
                            end_measurement_index=end,
                            anomalous=self._is_anomalous(start, index),
                        )
                    )

                    samples = self._load_raw_samples(
                        input_file_handle, metadata, start, end
                    )
                    output_file_handle.write(samples)
        metadata.length = len(sample_indices)
        _write_metadata_file(output_path, metadata)
        _write_index_file(output_path, index_entries)


if __name__ == "__main__":
    sampler = PBimSampler(128, True, MinutesPerHourSamplingStrategy(2))
    input_path = Path("../data/assembled/PBIM/N/april-week-01/assembled.dat")
    output_path = Path("/tmp/filtered/output.dat")
    sampler.process(input_path, output_path)
