import json
import math
import shutil
import struct
from pathlib import Path
from typing import Optional, List, BinaryIO

import click

from pbim_preprocessor.metadata import DatasetMetadata, _write_metadata_file
from pbim_preprocessor.index import _write_index, CutIndex
from pbim_preprocessor.merge import MergeConfig
from pbim_preprocessor.statistic import StatisticsCollector
from pbim_preprocessor.utils import LOGGER, _load_metadata


def _is_window_index_or_raise(indices: List[CutIndex]) -> bool:
    # Check that all indices are either windowed or none are
    if any(index.is_window_index for index in indices) and not all(
        index.is_window_index for index in indices
    ):
        raise ValueError("All indices must be windowed or none must be windowed")
    return indices[0].is_window_index


def _find_start_path(data_directory: Path) -> Path:
    names = sorted(
        [p.name for p in data_directory.iterdir() if p.is_dir()], key=lambda x: int(x)
    )
    if len(names) == 0:
        return next(data_directory.glob("*.dat"))
    return _find_start_path(data_directory / names[0])


def _find_next_path(data_directory: Path, current: Path) -> Optional[Path]:
    year, month = int(current.parent.name), int(current.name)
    if month == 12:
        year += 1
        month = 1
    else:
        month += 1
    next_path = data_directory / str(year) / str(month)
    if next_path.exists():
        f = [p for p in next_path.iterdir() if p.is_file() and p.name.endswith(".dat")]
        if len(f) > 0:
            return f[0]
        return _find_next_path(data_directory, next_path)
    else:
        return None


def _load_index(path: Path) -> CutIndex:
    index_path = path.parent / f"{path.stem}.index"
    with open(index_path, "rb") as f:
        return CutIndex.from_bytes(f.read())


def _parse_values(data: bytes, time_byte_size: int, channels: List[str]) -> List[float]:
    time_format = "q" if time_byte_size == 8 else "i"
    format_string = f"<{time_format}{'f' * (len(channels) - 1)}"
    return [*struct.unpack(format_string, data)]


CHUNK_SIZE = 8 * 1024 * 1024


def _estimate_steps(
    f: BinaryIO, chunk_size: int, ratio: float = 1.0, offset: float = 0
) -> int:
    f.seek(0, 2)
    size = f.tell()
    f.seek(0)
    actual_size = (1 - offset) * size

    return math.ceil(actual_size * ratio / chunk_size)


def _write_file_contiguous(
    input_file_path: Path,
    output_file_handle: BinaryIO,
    metadata: DatasetMetadata,
    statistics_collector: Optional[StatisticsCollector] = None,
    offset: float = 0.0,
    ratio: float = 1.0,
    include_in_statistics: bool = True,
):
    assert offset + ratio <= 1.0
    num_measurements = 0
    with open(input_file_path, "rb") as f:
        steps = _estimate_steps(f, CHUNK_SIZE, ratio, offset)
        LOGGER.info(f"Processing {input_file_path} (Estimated steps: {steps})")
        i = 0
        f.seek(0, 2)
        size = f.tell()
        total_num_measurements = size // metadata.measurement_size_in_bytes
        f.seek(
            int(total_num_measurements * offset) * metadata.measurement_size_in_bytes
        )
        measurements_to_write = int(total_num_measurements * ratio)

        left_over = b""
        while True:
            new_data = f.read(CHUNK_SIZE)
            chunk = left_over + new_data
            if chunk:
                LOGGER.info(f"Processing chunk {i + 1}/{steps} of {input_file_path}")
                measurements_in_chunk = len(chunk) // metadata.measurement_size_in_bytes
                if num_measurements + measurements_in_chunk > measurements_to_write:
                    chunk = chunk[
                        : (measurements_to_write - num_measurements)
                        * metadata.measurement_size_in_bytes
                    ]
                    measurements_in_chunk = measurements_to_write - num_measurements
                output_file_handle.write(
                    chunk[: measurements_in_chunk * metadata.measurement_size_in_bytes]
                )
                num_measurements += measurements_in_chunk
                left_over = chunk[
                    measurements_in_chunk * metadata.measurement_size_in_bytes :
                ]
                if statistics_collector is not None and include_in_statistics:
                    for step in range(measurements_in_chunk):
                        start = step * metadata.measurement_size_in_bytes
                        end = (step + 1) * metadata.measurement_size_in_bytes
                        values = _parse_values(
                            chunk[start:end],
                            metadata.time_byte_size,
                            metadata.channel_order,
                        )

                        for channel, value in zip(metadata.channel_order, values):
                            statistics_collector.add(channel, value)
                if num_measurements >= measurements_to_write:
                    break
            else:
                break
            i += 1
        return num_measurements


def _write_file_windowed(
    input_file_path: Path,
    output_file_handle: BinaryIO,
    metadata: DatasetMetadata,
    index: CutIndex,
    statistics_collector: Optional[StatisticsCollector] = None,
    offset: float = 0.0,
    ratio: float = 1.0,
    include_in_statistics: bool = True,
):
    start_index_entry = int(offset * len(index.entries))
    stop_index_entry = int((offset + ratio) * len(index.entries))
    total_windows = stop_index_entry - start_index_entry
    with open(input_file_path, "rb") as f:
        steps = _estimate_steps(f, CHUNK_SIZE, ratio, offset)
        LOGGER.info(f"Processing {input_file_path} (Estimated steps: {steps})")
        i = 0
        f.seek(0, 2)
        start_measurement_index = index.entries[
            start_index_entry
        ].start_measurement_index
        end_measurement_index = index.entries[
            stop_index_entry - 1
        ].end_measurement_index
        f.seek(start_measurement_index * metadata.measurement_size_in_bytes)
        measurements_to_write = end_measurement_index - start_measurement_index
        num_measurements = 0
        left_over = b""
        while True:
            new_data = f.read(CHUNK_SIZE)
            chunk = left_over + new_data
            if chunk:
                LOGGER.info(f"Processing chunk {i + 1}/{steps} of {input_file_path}")
                measurements_in_chunk = len(chunk) // metadata.measurement_size_in_bytes
                if num_measurements + measurements_in_chunk > measurements_to_write:
                    chunk = chunk[
                        : (measurements_to_write - num_measurements)
                        * metadata.measurement_size_in_bytes
                    ]
                    measurements_in_chunk = measurements_to_write - num_measurements
                output_file_handle.write(
                    chunk[: measurements_in_chunk * metadata.measurement_size_in_bytes]
                )
                num_measurements += measurements_in_chunk
                left_over = chunk[
                    measurements_in_chunk * metadata.measurement_size_in_bytes :
                ]
                if statistics_collector is not None and include_in_statistics:
                    for step in range(measurements_in_chunk):
                        start = step * metadata.measurement_size_in_bytes
                        end = (step + 1) * metadata.measurement_size_in_bytes
                        values = _parse_values(
                            chunk[start:end],
                            metadata.time_byte_size,
                            metadata.channel_order,
                        )

                        for channel, value in zip(metadata.channel_order, values):
                            statistics_collector.add(channel, value)
                if num_measurements >= measurements_to_write:
                    break
            else:
                break
            i += 1
    return total_windows


def _write_file(
    input_file_path: Path,
    output_file_handle: BinaryIO,
    metadata: DatasetMetadata,
    index: CutIndex,
    statistics_collector: Optional[StatisticsCollector] = None,
    offset: float = 0.0,
    ratio: float = 1.0,
    include_in_statistics: bool = True,
) -> int:
    assert offset + ratio <= 1.0
    if index.is_window_index:
        return _write_file_windowed(
            input_file_path,
            output_file_handle,
            metadata,
            index,
            statistics_collector,
            offset,
            ratio,
            include_in_statistics,
        )
    else:
        return _write_file_contiguous(
            input_file_path,
            output_file_handle,
            metadata,
            statistics_collector,
            offset,
            ratio,
            include_in_statistics,
        )


def _merge_predefined_files(
    config: MergeConfig,
    output_file_handle: BinaryIO,
) -> List[int]:
    _validate_config(config)
    num_measurements_or_windows = []
    statistics_collector = (
        StatisticsCollector()
        if not config.keep_statistics and config.use_statistics_from is None
        else None
    )
    if config.use_statistics_from is not None:
        config.use_statistics_from = Path(config.use_statistics_from)
        metadata_path = config.use_statistics_from
    else:
        metadata_path = config.base_path / config.files[0].relative_path
    metadata = _load_metadata(metadata_path)
    indices = [
        _load_index(config.base_path / file.relative_path) for file in config.files
    ]
    is_window_indices = _is_window_index_or_raise(indices)
    lengths = [
        _load_metadata(config.base_path / file.relative_path).length
        for file in config.files
    ]
    for file, index in zip(config.files, indices):
        num_measurements_or_windows += [
            _write_file(
                config.base_path / file.relative_path,
                output_file_handle,
                metadata,
                index,
                statistics_collector,
                file.offset,
                file.ratio,
                file.include_in_statistics,
            )
        ]
    if is_window_indices:
        length = 0
        for i, num_windows in enumerate(num_measurements_or_windows):
            offset, ratio, index, = (
                config.files[i].offset,
                config.files[i].ratio,
                indices[i],
            )
            start_index_entry = index.entries[int(offset * len(index.entries))]
            stop_index_entry = index.entries[
                int((offset + ratio) * len(index.entries)) - 1
            ]
            length += (
                stop_index_entry.end_measurement_index
                - start_index_entry.start_measurement_index
            )
    else:
        length = sum(num_measurements_or_windows)
    _write_metadata(config, metadata, length, statistics_collector)
    anomalous = [file.is_anomalous for file in config.files] if config.files else False
    _write_index(
        config.output_file,
        is_window_indices,
        num_measurements_or_windows,
        anomalous,
        lengths,
        existing_indices=indices,
        offsets=[file.offset for file in config.files] if config.files else None,
    )
    return num_measurements_or_windows


def _merge_files_by_date():
    raise NotImplementedError()


def _write_metadata(
    config: MergeConfig,
    metadata: DatasetMetadata,
    num_measurements: int,
    statistics_collector: Optional[StatisticsCollector],
):
    metadata.length = num_measurements
    if statistics_collector is not None:
        metadata.statistics = statistics_collector.get_all_channel_statistics()
    _write_metadata_file(config.output_file, metadata)


def _validate_config(config: MergeConfig) -> bool:
    files_specified = config.files is not None
    date_specified = config.start is not None and config.end is not None
    if files_specified and date_specified:
        raise ValueError("Cannot specify both files and date range")
    if not files_specified and not date_specified:
        raise ValueError("Must specify either files or date range")
    if config.use_statistics_from is not None and config.keep_statistics:
        raise ValueError("Cannot use statistics from file and keep statistics")

    return files_specified


@click.command()
@click.argument(
    "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def merge(
    config_path: Path,
):
    with open(config_path, "r") as f:
        config = MergeConfig.from_dict(json.load(f))
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    if _validate_config(config):
        with open(config.output_file, "wb") as f:
            _merge_predefined_files(config, f)
    else:
        _merge_files_by_date()
