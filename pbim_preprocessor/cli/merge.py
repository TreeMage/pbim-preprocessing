import json
import struct
from pathlib import Path
from typing import Optional, List

import click

from pbim_preprocessor.cli.assemble import DatasetMetadata
from pbim_preprocessor.statistics import StatisticsCollector


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
    return next((data_directory / str(year) / str(month).zfill(2)).glob("*.dat"))


def _load_metadata(path: Path) -> DatasetMetadata:
    meta_data_file = next(path.parent.glob("*.json"))
    with open(meta_data_file, "r") as f:
        return DatasetMetadata.from_dict(json.load(f))


def _parse_values(data: bytes) -> List[float]:
    return [struct.unpack("<f", data[i : i + 4])[0] for i in range(4, len(data), 4)]


CHUNK_SIZE = 1024 * 1024


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument(
    "output-file", type=click.Path(writable=True, dir_okay=False, path_type=Path)
)
def merge(path: Path, output_file: Path):
    statistics_collector = StatisticsCollector()
    current_path = _find_start_path(path)
    metadata = _load_metadata(current_path)
    with open(output_file, "wb") as f:
        while current_path.exists():
            with open(current_path, "rb") as current_file:
                while True:
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
            current_path = _find_next_path(path, current_path)
    with open(output_file.parent / f"{output_file.stem}.metadata.json", "w") as f:
        json.dump(
            metadata.to_dict(),
            f,
            indent=4,
        )
