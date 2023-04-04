import abc
import calendar
import datetime
import struct
from pathlib import Path
from typing import List, BinaryIO, Optional, Tuple

import numpy as np
import tqdm

from pbim_preprocessor.cli.merge import _load_index
from pbim_preprocessor.index import CutIndexEntry, _write_index_file
from pbim_preprocessor.metadata import DatasetMetadata, _write_metadata_file
from pbim_preprocessor.utils import _load_metadata, LOGGER


def _find_group_index(index: int, start_and_end_indices: List[Tuple[int, int]]):
    for i, (start, end) in enumerate(start_and_end_indices):
        if start <= index < end:
            return i

    end_start_pairs = [(0, start_and_end_indices[0][0])]
    for i in range(len(start_and_end_indices) - 1):
        first_start, first_end = start_and_end_indices[i]
        second_start, second_end = start_and_end_indices[i + 1]
        end_start_pairs.append((first_end, second_start))

    for i, (end, start) in enumerate(end_start_pairs):
        if end <= index < start:
            return i

    return None


def _sample_next_interval(
    time: np.ndarray,
    sample_length_in_seconds: float,
    start_and_end_indices: List[Tuple[int, int]],
    start_index: int,
):
    current_length = 0

    if not (group_index := _find_group_index(start_index, start_and_end_indices)):
        return None

    sample_indices = []
    current_start, current_end = start_and_end_indices[group_index]
    if start_index < current_start:
        start_index = current_start
    current_index = start_index
    while current_length < sample_length_in_seconds:
        sample_indices.append(current_index)
        current_length = (time[current_index] - time[start_index]) / 1000
        current_index += 1
        if current_index >= current_end:
            group_index += 1
            if group_index >= len(start_and_end_indices):
                break
            current_start, current_end = start_and_end_indices[group_index]
            current_index = current_start

    return sample_indices, current_index


class DatasetSamplingStrategy(abc.ABC):
    @abc.abstractmethod
    def compute_sample_indices(
        self, time: np.ndarray, start_and_end_indices: List[Tuple[int, int]]
    ) -> List[int]:
        pass


class NoSamplingStrategy(DatasetSamplingStrategy):
    def compute_sample_indices(
        self, time: np.ndarray, start_and_end_indices: List[Tuple[int, int]]
    ) -> List[int]:
        final_indices = []
        for start, end in start_and_end_indices:
            final_indices.extend(list(range(start, end)))
        return final_indices


class UniformSamplingStrategy(DatasetSamplingStrategy):
    def __init__(self, num_samples: int, window_size: int):
        self._num_samples = num_samples
        self._window_size = window_size

    def compute_sample_indices(
        self, time: np.ndarray, start_and_end_indices: List[Tuple[int, int]]
    ) -> List[int]:
        available_samples = sum(end - start for start, end in start_and_end_indices)
        if available_samples < self._num_samples:
            raise ValueError(
                f"Cannot sample {self._num_samples} from {available_samples} samples."
            )
        steps = available_samples // self._num_samples
        final_indices = []
        for start, end in start_and_end_indices:
            for step in range(start, end, steps):
                final_indices.extend([step + i for i in range(self._window_size)])
        return final_indices


class RandomSamplingStrategy(DatasetSamplingStrategy):
    def __init__(self, num_samples: int, window_size: int):
        self._num_samples = num_samples
        self._window_size = window_size

    def compute_sample_indices(
        self, time: np.ndarray, start_and_end_indices: List[Tuple[int, int]]
    ) -> List[int]:
        available_samples = sum(end - start for start, end in start_and_end_indices)
        if available_samples < self._num_samples:
            raise ValueError(
                f"Cannot sample {self._num_samples} from {available_samples} samples."
            )
        window_indices = []
        for start, end in start_and_end_indices:
            window_indices.extend(list(range(start, end)))
        sampled_indices = np.random.choice(
            window_indices, self._num_samples, replace=False
        ).tolist()
        final_indices = []
        for index in sampled_indices:
            final_indices.extend([index + i for i in range(self._window_size)])
        return final_indices


class WeightedRandomSamplingStrategy(DatasetSamplingStrategy):
    def __init__(self, num_samples: int, window_size: int):
        self._num_samples = num_samples
        self._window_size = window_size

    @staticmethod
    def _compute_weight(timestamp: int) -> float:
        time = datetime.datetime.utcfromtimestamp(timestamp / 1000)
        if 6 <= time.hour <= 9:
            return 1
        if 16 <= time.hour <= 19:
            return 1
        if 20 <= time.hour <= 5:
            return 0.25
        return 0.5

    def compute_sample_indices(
        self, time: np.ndarray, start_and_end_indices: List[Tuple[int, int]]
    ) -> List[int]:
        available_samples = sum(end - start for start, end in start_and_end_indices)
        if available_samples < self._num_samples:
            raise ValueError(
                f"Cannot sample {self._num_samples} from {available_samples} samples."
            )
        window_indices = []
        for start, end in start_and_end_indices:
            window_indices.extend(list(range(start, end)))
        weights = np.array(
            [self._compute_weight(time[index]) for index in window_indices]
        )
        weights = weights / np.sum(weights)
        sampled_indices = np.random.choice(
            window_indices, self._num_samples, replace=False, p=weights
        ).tolist()
        final_indices = []
        for index in sampled_indices:
            final_indices.extend([index + i for i in range(self._window_size)])
        return final_indices


class IntervalSamplingStrategy(DatasetSamplingStrategy):
    def __init__(
        self,
        interval_length_in_seconds: int,
        samples_per_interval: int,
        sample_length_in_seconds: int,
    ):
        self._interval_length_in_seconds = interval_length_in_seconds
        self._samples_per_interval = samples_per_interval
        self._sample_length_in_seconds = sample_length_in_seconds

    @staticmethod
    def _make_datetime(timestamp: float) -> datetime.datetime:
        return datetime.datetime.utcfromtimestamp(timestamp / 1000)

    @staticmethod
    def _make_timestamp(dt: datetime.datetime) -> float:
        return calendar.timegm(dt.timetuple()) * 1000

    def compute_sample_indices(
        self, time: np.ndarray, start_and_end_indices: List[Tuple[int, int]]
    ) -> List[int]:
        final_indices = []
        start_date = self._make_datetime(time[0])
        end_date = self._make_datetime(time[-1])
        current_date = start_date
        while current_date < end_date:
            for i in range(self._samples_per_interval):
                interval_start = current_date + datetime.timedelta(
                    seconds=i
                    * self._interval_length_in_seconds
                    / self._samples_per_interval
                )
                if interval_start > end_date:
                    break
                if len(final_indices) > 0 and interval_start < self._make_datetime(
                    time[final_indices[-1]]
                ):
                    LOGGER.warn(
                        f"Skipping interval start {interval_start} because it is before the last sample."
                    )
                    continue
                start_index = np.searchsorted(
                    time, self._make_timestamp(interval_start)
                ).item()
                interval = _sample_next_interval(
                    time,
                    self._sample_length_in_seconds,
                    start_and_end_indices,
                    start_index,
                )
                if not interval:
                    break
                sample_indices, end_index = interval
                final_indices.extend(sample_indices)
            current_date += datetime.timedelta(seconds=self._interval_length_in_seconds)
        return final_indices


class HourlySamplingStrategy(IntervalSamplingStrategy):
    def __init__(self, samples_per_hour: int, sample_length_in_seconds: int):
        super().__init__(3600, samples_per_hour, sample_length_in_seconds)


class MinutelySamplingStrategy(IntervalSamplingStrategy):
    def __init__(self, samples_per_minute: int, sample_length_in_seconds: int):
        super().__init__(60, samples_per_minute, sample_length_in_seconds)


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

    @staticmethod
    def _parse_samples(bytes: bytes, metadata: DatasetMetadata) -> np.ndarray:
        time_byte_format = "<q" if metadata.time_byte_size == 8 else "<i"
        format_string = time_byte_format + "f" * (len(metadata.channel_order) - 1)
        return np.array(
            [
                struct.unpack(
                    format_string,
                    bytes[i : i + metadata.measurement_size_in_bytes],
                )
                for i in range(0, len(bytes), metadata.measurement_size_in_bytes)
            ]
        )

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
                (
                    int(indices[current_start_index]),
                    int(indices[min(current_end_index, len(indices) - 1)] + 1),
                )
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
        indices = []
        with open(input_path, "rb") as f:
            for i in tqdm.trange(number_of_windows, desc="Loading windows"):
                window = self._load_window(f, i, metadata)
                window_without_time = window[1:, :]
                if self._remove_zero_windows and np.all(window_without_time == 0):
                    continue
                indices.append(i)
            time = self._load_time(f, metadata)
        contiguous_start_end_indices = self._compute_start_and_end_indices(indices)
        sample_indices = self._sampling_strategy.compute_sample_indices(
            time, contiguous_start_end_indices
        )
        index_entries = []
        with open(output_path, "wb") as output_file_handle:
            with open(input_path, "rb") as input_file_handle:
                contiguous_start_end_sample_indices = self._compute_start_and_end_indices(sample_indices)
                for start, end in tqdm.tqdm(
                    contiguous_start_end_sample_indices,
                    desc="Writing continuous sample chunks",
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
        metadata.length = sum([end - start for start, end in contiguous_start_end_sample_indices])
        _write_metadata_file(output_path, metadata)
        _write_index_file(output_path, index_entries)
