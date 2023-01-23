import datetime
import json
import struct
from pathlib import Path
from typing import Optional, List, BinaryIO

import click

from pbim_preprocessor.cli.assemble import DatasetMetadata
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
    meta_data_file = next(path.parent.glob("*.json"))
    with open(meta_data_file, "r") as f:
        return DatasetMetadata.from_dict(json.load(f))


def _parse_values(data: bytes) -> List[float]:
    return [struct.unpack("<f", data[i : i + 4])[0] for i in range(4, len(data), 4)]


CHUNK_SIZE = 1024 * 1024


def _estimate_steps(f: BinaryIO, chunk_size: int) -> int:
    f.seek(0, 2)
    size = f.tell()
    f.seek(0)
    return size // chunk_size


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument(
    "output-file", type=click.Path(writable=True, dir_okay=False, path_type=Path)
)
@click.option("--start", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", type=click.DateTime(formats=["%Y-%m-%d"]))
def merge(
    path: Path,
    output_file: Path,
    start: Optional[datetime.datetime] = None,
    end: Optional[datetime.datetime] = None,
):
    statistics_collector = StatisticsCollector()
    current_path = (
        _find_start_path(path)
        if start is None
        else next((path / str(start.year) / str(start.month).zfill(2)).glob("*.dat"))
    )
    metadata = _load_metadata(current_path)
    with open(output_file, "wb") as f:
        while current_path.exists():
            with open(current_path, "rb") as current_file:
                steps = _estimate_steps(current_file, CHUNK_SIZE)
                LOGGER.info(
                    f"Processing {current_path.parent.name}/{current_path.name} (Estimated steps: {steps})"
                )
                i = 0
                while True:
                    LOGGER.info(
                        f"Processing chunk {i + 1}/{steps} of {current_path.parent.name}/{current_path.name}"
                    )
                    chunk = current_file.read(CHUNK_SIZE)
                    if chunk:
                        f.write(chunk)
                        for step in range(
                            len(chunk) // metadata.measurement_size_in_bytes
                        ):
                            start = step * metadata.measurement_size_in_bytes + 4
                            end = (step + 1) * metadata.measurement_size_in_bytes
                            values = _parse_values(chunk[start:end])
                            for channel, value in zip(metadata.channel_order, values):
                                statistics_collector.add(channel, value)
                    else:
                        break
                    i += 1
            current_path = _find_next_path(path, current_path)
            if (
                end is not None
                and current_path is not None
                and current_path.parent.parent.name == str(end.year)
                and current_path.parent.name == str(end.month).zfill(2)
            ):
                break
    with open(output_file.parent / f"{output_file.stem}.metadata.json", "w") as f:
        json.dump(
            metadata.to_dict(),
            f,
            indent=4,
        )
