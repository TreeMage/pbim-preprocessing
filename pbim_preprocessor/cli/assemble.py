import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import click

from pbim_preprocessor.assembler import Assembler
from pbim_preprocessor.parser import POST_PROCESSABLE_CHANNELS
from pbim_preprocessor.processor import MEASUREMENT_SIZE_IN_BYTES
from pbim_preprocessor.sampling import (
    MeanSamplingStrategy,
    LinearInterpolationSamplingStrategy,
)
from pbim_preprocessor.writer import CsvWriter, BinaryWriter

STRATEGIES = {
    "mean": MeanSamplingStrategy(),
    "interpolate": LinearInterpolationSamplingStrategy(),
}
FORMATS = {
    "csv": CsvWriter,
    "binary": BinaryWriter,
}

CHANNELS = POST_PROCESSABLE_CHANNELS


@dataclass
class DatasetMetadata:
    channel_order: List[str]
    start_time: datetime.datetime
    end_time: datetime.datetime
    measurement_size_in_bytes: int
    resolution: int
    length: int


def _write_metadata_file(path: Path, metadata: DatasetMetadata):
    metadata_path = path.parent / f"{path.stem}.metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "channel_order": metadata.channel_order,
                "start_time": metadata.start_time.timestamp(),
                "end_time": metadata.end_time.timestamp(),
                "measurement_size_in_bytes": metadata.measurement_size_in_bytes,
                "resolution": metadata.resolution,
                "length": metadata.length,
            },
            f,
            indent=4,
        )


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument(
    "output-path", type=click.Path(writable=True, dir_okay=False, path_type=Path)
)
@click.argument("start-time", type=click.DateTime())
@click.argument("end-time", type=click.DateTime())
@click.argument("resolution", type=click.INT)
@click.option("--strategy", default="mean", type=click.Choice(list(STRATEGIES.keys())))
@click.option("--output-format", default="csv", type=click.Choice(list(FORMATS.keys())))
def assemble(
    path: Path,
    output_path: Path,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    resolution: int,
    strategy: str,
    output_format: str,
):
    output_path.parent.mkdir(exist_ok=True, parents=True)
    assembler = Assembler(STRATEGIES[strategy], resolution)
    writer_type = FORMATS[output_format]
    with writer_type(output_path, CHANNELS) as writer:
        length = 0
        for step in assembler.assemble(
            path, start_time=start_time, end_time=end_time, channels=CHANNELS
        ):
            time = int(step["time"])
            writer.write_step(step, time)
            length += 1

    _write_metadata_file(
        output_path,
        DatasetMetadata(
            channel_order=["Time"] + CHANNELS,
            start_time=start_time,
            end_time=end_time,
            measurement_size_in_bytes=MEASUREMENT_SIZE_IN_BYTES,
            resolution=resolution,
            length=length,
        ),
    )
