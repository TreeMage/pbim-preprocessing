import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import click
from dataclasses_json import dataclass_json

from pbim_preprocessor.assembler import Assembler
from pbim_preprocessor.parser import POST_PROCESSABLE_CHANNELS
from pbim_preprocessor.sampling import (
    MeanSamplingStrategy,
    LinearInterpolationSamplingStrategy,
)
from pbim_preprocessor.statistics import StatisticsCollector, ChannelStatistics
from pbim_preprocessor.utils import LOGGER
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


@dataclass_json
@dataclass
class DatasetMetadata:
    channel_order: List[str]
    start_time: int
    end_time: int
    measurement_size_in_bytes: int
    resolution: int
    length: int
    statistics: Dict[str, ChannelStatistics]


def _write_metadata_file(path: Path, metadata: DatasetMetadata):
    metadata_path = path.parent / f"{path.stem}.metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(
            metadata.to_dict(),
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
@click.option("--channel", default=CHANNELS, multiple=True)
@click.option("--debug", is_flag=True, default=False)
def assemble(
    path: Path,
    output_path: Path,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    resolution: int,
    strategy: str,
    output_format: str,
    channel: List[str],
    debug: bool,
):
    LOGGER.set_debug(debug)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    channels = list(channel)
    assembler = Assembler(STRATEGIES[strategy], resolution)
    writer_type = FORMATS[output_format]
    statistics_collector = StatisticsCollector()
    with writer_type(output_path, channels) as writer:
        length = 0
        for step in assembler.assemble(
            path, start_time=start_time, end_time=end_time, channels=channels
        ):
            time = int(step["time"])
            statistics_collector.add_all(step)
            writer.write_step(step, time)
            length += 1

    _write_metadata_file(
        output_path,
        DatasetMetadata(
            channel_order=["Time"] + channels,
            start_time=int(start_time.timestamp()),
            end_time=int(end_time.timestamp()),
            # Time + Channels
            measurement_size_in_bytes=4 + len(channels) * 4,
            resolution=resolution,
            length=length,
            statistics=statistics_collector.get_all_channel_statistics(),
        ),
    )
