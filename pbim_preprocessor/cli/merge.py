import json
import math
import struct
from pathlib import Path
from typing import Optional, List, BinaryIO

import click

from pbim_preprocessor.cli.assemble import DatasetMetadata
from pbim_preprocessor.index import _write_index, CutIndexEntry
from pbim_preprocessor.merge import MergeConfig
from pbim_preprocessor.statistic import StatisticsCollector
from pbim_preprocessor.utils import LOGGER


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


def _load_metadata(path: Path) -> DatasetMetadata:
    meta_data_file = path.parent / f"{path.stem}.metadata.json"
    with open(meta_data_file, "r") as f:
        return DatasetMetadata.from_dict(json.load(f))


def _load_index(path: Path) -> List[CutIndexEntry]:
    index_path = path.parent / f"{path.stem}.index.json"
    with open(index_path, "r") as f:
        return [CutIndexEntry.from_dict(entry) for entry in json.load(f)]


def _parse_values(data: bytes) -> List[float]:
    return [struct.unpack("<f", data[i : i + 4])[0] for i in range(0, len(data), 4)]


CHUNK_SIZE = 1024 * 1024


def _estimate_steps(f: BinaryIO, chunk_size: int, ratio: float = 1.0) -> int:
    f.seek(0, 2)
    size = f.tell()
    f.seek(0)

    return math.ceil(size * ratio / chunk_size)


def _write_file(
    input_file_path: Path,
    output_file_handle: BinaryIO,
    metadata: DatasetMetadata,
    statistics_collector: Optional[StatisticsCollector] = None,
    offset: float = 0.0,
    ratio: float = 1.0,
) -> int:
    assert offset + ratio <= 1.0
    num_measurements = 0
    with open(input_file_path, "rb") as f:
        steps = _estimate_steps(f, CHUNK_SIZE, ratio)
        LOGGER.info(f"Processing {input_file_path} (Estimated steps: {steps})")
        i = 0
        f.seek(0, 2)
        size = f.tell()
        total_num_measurements = size // metadata.measurement_size_in_bytes
        f.seek(
            int(total_num_measurements * offset) * metadata.measurement_size_in_bytes
        )
        measurements_to_write = int(total_num_measurements * ratio)

        while True:
            chunk = f.read(CHUNK_SIZE)
            if chunk:
                LOGGER.info(f"Processing chunk {i + 1}/{steps} of {input_file_path}")
                measurements_in_chunk = len(chunk) // metadata.measurement_size_in_bytes
                if num_measurements + measurements_in_chunk > measurements_to_write:
                    chunk = chunk[
                        : (measurements_to_write - num_measurements)
                        * metadata.measurement_size_in_bytes
                    ]
                    measurements_in_chunk = measurements_to_write - num_measurements
                output_file_handle.write(chunk)
                num_measurements += measurements_in_chunk
                if statistics_collector is not None:
                    for step in range(measurements_in_chunk):
                        start = step * metadata.measurement_size_in_bytes
                        end = (step + 1) * metadata.measurement_size_in_bytes
                        values = _parse_values(chunk[start:end])
                        for channel, value in zip(metadata.channel_order, values):
                            statistics_collector.add(channel, value)
                if num_measurements >= measurements_to_write:
                    break
            else:
                break
            i += 1
        return num_measurements


def _merge_predefined_files(
    config: MergeConfig,
    output_file_handle: BinaryIO,
) -> List[int]:
    num_measurements = []
    statistics_collector = StatisticsCollector() if not config.keep_statistics else None
    metadata = _load_metadata(config.base_path / config.files[0].relative_path)
    indices = [
        _load_index(config.base_path / file.relative_path) for file in config.files
    ]
    for file in config.files:
        num_measurements += [
            _write_file(
                config.base_path / file.relative_path,
                output_file_handle,
                metadata,
                statistics_collector,
                file.offset,
                file.ratio,
            )
        ]
    _write_metadata(config, metadata, sum(num_measurements), statistics_collector)
    anomalous = [file.is_anomalous for file in config.files] if config.files else False
    _write_index(
        num_measurements,
        anomalous,
        config.output_file.parent / f"{config.output_file.stem}.index.json",
        exsiting_indices=indices,
    )
    return num_measurements


def _merge_files_by_date():
    raise NotImplementedError()


def _write_metadata(
    config: MergeConfig,
    metadata: DatasetMetadata,
    num_measurements: int,
    statistics_collector: Optional[StatisticsCollector],
):
    metadata.length = num_measurements
    if not config.keep_statistics:
        metadata.statistics = statistics_collector.get_all_channel_statistics()
    with open(
        config.output_file.parent / f"{config.output_file.stem}.metadata.json", "w"
    ) as f:
        json.dump(metadata.to_dict(), f, indent=4)


def _validate_config(config: MergeConfig) -> bool:
    files_specified = config.files is not None
    date_specified = config.start is not None and config.end is not None
    if files_specified and date_specified:
        raise ValueError("Cannot specify both files and date range")
    if not files_specified and not date_specified:
        raise ValueError("Must specify either files or date range")
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
