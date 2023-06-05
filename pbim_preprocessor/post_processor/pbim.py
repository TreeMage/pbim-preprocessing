import abc
import functools
import shutil
import struct
from pathlib import Path
from typing import List, BinaryIO, Optional, Tuple, Callable

import numpy as np
import tqdm

from pbim_preprocessor.cli.merge import _load_index
from pbim_preprocessor.index import CutIndexEntry, _write_index_file, CutIndex
from pbim_preprocessor.metadata import DatasetMetadata, _write_metadata_file
from pbim_preprocessor.post_processor.sampling import DatasetSamplingStrategy
from pbim_preprocessor.utils import _load_metadata


EXCLUDE_CHANNELS = ["Time", "Temperature"]


class DatasetSampler:
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
    def _parse_samples(data: bytes, metadata: DatasetMetadata) -> np.ndarray:
        time_byte_format = "<q" if metadata.time_byte_size == 8 else "<i"
        format_string = time_byte_format + "f" * (len(metadata.channel_order) - 1)
        return np.array(
            [
                struct.unpack(
                    format_string,
                    data[i : i + metadata.measurement_size_in_bytes],
                )
                for i in range(0, len(data), metadata.measurement_size_in_bytes)
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
        start = 0
        for i in range(1, len(indices)):
            if indices[i] != indices[i - 1] + 1:
                if start < i - 1:
                    computed_indices.append((indices[start], indices[i - 1] + 1))
                start = i
        if start < len(indices) - 1:
            computed_indices.append((indices[start], indices[-1] + 1))
        return computed_indices

    @staticmethod
    def _is_anomalous(index: int, cut_index: CutIndex) -> bool:
        for entry in cut_index.entries:
            if entry.start_measurement_index <= index < entry.end_measurement_index:
                return entry.anomalous
        raise ValueError(f"Index {index} is not in the cut index")

    @staticmethod
    def _find_index_entry_for_index(index: CutIndex, i: int) -> CutIndexEntry:
        for entry in index.entries:
            if entry.start_measurement_index <= i < entry.end_measurement_index:
                return entry

        raise ValueError(f"Failed to determine index entry for index {i}.")

    def process(self, input_path: Path, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = _load_metadata(input_path)
        index = _load_index(input_path)
        number_of_windows = metadata.length - self._window_size + 1
        indices = []
        exclude_indices = [
            metadata.channel_order.index(channel) for channel in EXCLUDE_CHANNELS
        ]
        with open(input_path, "rb") as f:
            for i in tqdm.trange(number_of_windows, desc="Loading windows"):
                index_entry = self._find_index_entry_for_index(index, i)
                if i + self._window_size >= index_entry.end_measurement_index:
                    continue
                window = self._load_window(f, i, metadata)
                window = np.delete(window, exclude_indices, axis=0)
                if self._remove_zero_windows and np.all(window == 0):
                    continue
                indices.append(i)
            time = self._load_time(f, metadata)
        contiguous_start_end_indices = self._compute_start_and_end_indices(indices)
        sample_indices = self._sampling_strategy.compute_sample_indices(
            time, contiguous_start_end_indices
        )
        print("Copying data...")
        shutil.copyfile(input_path, output_path)
        index_entries = []
        num_measurements = 0
        for start, end in tqdm.tqdm(
            sample_indices,
            desc="Writing index entries",
        ):
            index_entries.append(
                CutIndexEntry(
                    start_measurement_index=start,
                    end_measurement_index=end,
                    anomalous=self._is_anomalous(start, index),
                )
            )
            num_measurements += end - start
        metadata.length = num_measurements
        _write_metadata_file(output_path, metadata)
        _write_index_file(output_path, CutIndex(True, index_entries))


class BaseDatasetSampler(abc.ABC):
    def __init__(
        self,
        window_size: int,
        sampling_strategy: DatasetSamplingStrategy,
        check_window_validity: bool,
        window_validity_check_predicate: Callable[[np.ndarray], bool],
        channels_excluded_in_window_validity_check: List[str],
        produces_windowed_index: bool,
    ):
        self._window_size = window_size
        self._sampling_strategy = sampling_strategy
        self._check_window_validity = check_window_validity
        self._window_validity_check_predicate = window_validity_check_predicate
        self._channels_excluded_in_window_validity_check = (
            channels_excluded_in_window_validity_check
        )
        self._produces_windowed_index = produces_windowed_index

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
    def _parse_samples(data: bytes, metadata: DatasetMetadata) -> np.ndarray:
        time_byte_format = "<q" if metadata.time_byte_size == 8 else "<i"
        format_string = time_byte_format + "f" * (len(metadata.channel_order) - 1)
        return np.array(
            [
                struct.unpack(
                    format_string,
                    data[i : i + metadata.measurement_size_in_bytes],
                )
                for i in range(0, len(data), metadata.measurement_size_in_bytes)
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
        start = 0
        for i in range(1, len(indices)):
            if indices[i] != indices[i - 1] + 1:
                if start < i - 1:
                    computed_indices.append((indices[start], indices[i - 1] + 1))
                start = i
        if start < len(indices) - 1:
            computed_indices.append((indices[start], indices[-1] + 1))
        return computed_indices

    @staticmethod
    def _is_anomalous(index: int, cut_index: CutIndex) -> bool:
        for entry in cut_index.entries:
            if entry.start_measurement_index <= index < entry.end_measurement_index:
                return entry.anomalous
        raise ValueError(f"Index {index} is not in the cut index")

    @staticmethod
    def _find_index_entry_for_index(index: CutIndex, i: int) -> CutIndexEntry:
        for entry in index.entries:
            if entry.start_measurement_index <= i < entry.end_measurement_index:
                return entry

        raise ValueError(f"Failed to determine index entry for index {i}.")

    def process(self, input_path: Path, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = _load_metadata(input_path)
        index = _load_index(input_path)
        number_of_windows = metadata.length - self._window_size + 1
        indices = []
        exclude_indices = [
            metadata.channel_order.index(channel)
            for channel in self._channels_excluded_in_window_validity_check
        ]
        with open(input_path, "rb") as f:
            for i in tqdm.trange(number_of_windows, desc="Loading windows"):
                index_entry = self._find_index_entry_for_index(index, i)
                if i + self._window_size >= index_entry.end_measurement_index:
                    continue
                if self._check_window_validity:
                    window = self._load_window(f, i, metadata)
                    window = np.delete(window, exclude_indices, axis=0)
                    if not self._window_validity_check_predicate(window):
                        continue
                indices.append(i)
            time = self._load_time(f, metadata)
        contiguous_start_end_indices = self._compute_start_and_end_indices(indices)
        sample_indices = self._sampling_strategy.compute_sample_indices(
            time, contiguous_start_end_indices
        )
        print("Copying data...")
        shutil.copyfile(input_path, output_path)
        index_entries = []
        num_measurements = 0
        for start, end in tqdm.tqdm(
            sample_indices,
            desc="Writing index entries",
        ):
            index_entries.append(
                CutIndexEntry(
                    start_measurement_index=start,
                    end_measurement_index=end,
                    anomalous=self._is_anomalous(start, index),
                )
            )
            num_measurements += end - start
        metadata.length = num_measurements
        _write_metadata_file(output_path, metadata)
        _write_index_file(
            output_path, CutIndex(self._produces_windowed_index, index_entries)
        )


class PBimDatasetSampler(BaseDatasetSampler):
    EXCLUDE_CHANNELS = ["Time", "Temperature"]

    def __init__(self, window_size: int, sampling_strategy: DatasetSamplingStrategy):
        super().__init__(
            window_size=window_size,
            sampling_strategy=sampling_strategy,
            check_window_validity=True,
            window_validity_check_predicate=lambda window: not np.all(window == 0),
            channels_excluded_in_window_validity_check=self.EXCLUDE_CHANNELS,
            produces_windowed_index=True,
        )


class LuxDatasetSampler(BaseDatasetSampler):
    # Exclude everything except the "Driving Point" (DP) sensor
    EXCLUDE_CHANNELS = [
        "Time",
        "30185",
        "21725",
        "30187",
        "29701",
        "30188",
        "29700",
        "30190",
        "21728",
        "30498",
        "21916",
        "30499",
        "21917",
        "30500",
        "21919",
        "30501",
        "30189",
        "39376",
        "39375",
        "39371",
        "39373",
        "39369",
        "39370",
        "39372",
        "30183",
        "29702",
        "Temperature",
    ]

    def __init__(
        self,
        window_size: int,
        sampling_rate: int,
        lower_frequency_bound: float,
        upper_frequency_bound: float,
        sampling_strategy: Optional[DatasetSamplingStrategy] = None,
    ):
        super().__init__(
            window_size=window_size,
            sampling_strategy=sampling_strategy,
            check_window_validity=True,
            window_validity_check_predicate=self._is_window_valid,
            channels_excluded_in_window_validity_check=self.EXCLUDE_CHANNELS,
            produces_windowed_index=True,
        )
        self._sampling_rate = sampling_rate
        self._lower_frequency_bound = lower_frequency_bound
        self._upper_frequency_bound = upper_frequency_bound

    def _is_window_valid(self, window: np.ndarray) -> bool:
        fft = np.fft.fft(window - np.mean(window))
        frequencies = np.fft.fftfreq(len(fft), 1 / self._sampling_rate)
        fft = fft[: len(fft) // 2]
        frequencies = frequencies[: len(frequencies) // 2]
        sorted_frequency_indices = np.argsort(np.abs(fft))[::-1]
        top_frequency = frequencies[sorted_frequency_indices[0]]
        return (
            self._lower_frequency_bound <= top_frequency <= self._upper_frequency_bound
        )
