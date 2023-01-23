import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import click
from dataclasses_json import dataclass_json

from pbim_preprocessor.assembler import (
    PBimAssembler,
    GrandStandAssembler,
    AssemblerWrapper,
)
from pbim_preprocessor.parser import POST_PROCESSABLE_CHANNELS
from pbim_preprocessor.sampling import (
    MeanSamplingStrategy,
    LinearInterpolationSamplingStrategy,
)
from pbim_preprocessor.statistic import StatisticsCollector, ChannelStatistics
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

CHANNELS = {
    "pbim": POST_PROCESSABLE_CHANNELS,
    "grandstand": [f"Joint {i}" for i in range(1, 30)],
}


@dataclass_json
@dataclass
class DatasetMetadata:
    channel_order: List[str]
    start_time: Optional[int]
    end_time: Optional[int]
    measurement_size_in_bytes: int
    resolution: Optional[int]
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


def _make_metadata(
    mode: str,
    channels: List[str],
    start_time: Optional[datetime.datetime],
    end_time: Optional[datetime.datetime],
    resolution: Optional[int],
    length: int,
    statistics: Dict[str, ChannelStatistics],
) -> DatasetMetadata:
    match mode:
        case "pbim":
            return DatasetMetadata(
                channel_order=["Time"] + channels,
                start_time=int(start_time.timestamp()) if start_time else None,
                end_time=int(end_time.timestamp()) if end_time else None,
                # Time + Channels
                measurement_size_in_bytes=4 + len(channels) * 4,
                resolution=resolution,
                length=length,
                statistics=statistics,
            )
        case "grandstand":
            return DatasetMetadata(
                channel_order=channels,
                start_time=None,
                end_time=None,
                # Channels
                measurement_size_in_bytes=len(channels) * 4,
                resolution=None,
                length=length,
                statistics=statistics,
            )


def _validate_args(
    mode: str,
    start_time: Optional[datetime.datetime],
    end_time: Optional[datetime.datetime],
    resolution: Optional[int],
    scenario: Optional[str],
):
    def _raise(parameter: str):
        raise click.BadParameter(f"Parameter {parameter} is required in mode {mode}.")

    match mode:
        case "pbim":
            if start_time is None:
                _raise("start_time")
            if end_time is None:
                _raise("end_time")
            if resolution is None:
                _raise("resolution")
        case "grandstand":
            if scenario is None:
                _raise("scenario")


def _prepare_channels(mode: str, channels: List[str]) -> List[str]:
    if not channels:
        return CHANNELS[mode]
    match mode:
        case "pbim":
            if "relevant" in channels:
                ignore_channels = ["MS17", "MS18", "MS19", "MS20", "MS21", "MS22"]
                channels = [c for c in CHANNELS["pbim"] if c not in ignore_channels]
            if "all" in channels:
                channels = CHANNELS["pbim"]
        case "grandstand":
            if "relevant" in channels or "all" in channels:
                channels = CHANNELS["grandstand"]
    return channels


@click.command()
@click.argument("mode", type=click.Choice(["pbim", "grandstand"]))
@click.option("--scenario", type=click.STRING)
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument(
    "output-path", type=click.Path(writable=True, dir_okay=False, path_type=Path)
)
@click.option("--start-time", default=None, type=click.DateTime())
@click.option("--end-time", default=None, type=click.DateTime())
@click.option("--resolution", default=None, type=click.INT)
@click.option("--strategy", default="mean", type=click.Choice(list(STRATEGIES.keys())))
@click.option("--output-format", default="csv", type=click.Choice(list(FORMATS.keys())))
@click.option("--channel", multiple=True)
@click.option("--debug", is_flag=True, default=False)
def assemble(
    mode: str,
    scenario: Optional[str],
    path: Path,
    output_path: Path,
    start_time: Optional[datetime.datetime],
    end_time: Optional[datetime.datetime],
    resolution: Optional[int],
    strategy: str,
    output_format: str,
    channel: List[str],
    debug: bool,
):
    LOGGER.set_debug(debug)
    _validate_args(mode, start_time, end_time, resolution, scenario)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    channels = _prepare_channels(mode, list(channel))
    assembler = AssemblerWrapper(
        mode,
        PBimAssembler(path, STRATEGIES[strategy], resolution)
        if mode == "pbim"
        else GrandStandAssembler(path),
    )
    writer_type = FORMATS[output_format]
    statistics_collector = StatisticsCollector()
    with writer_type(output_path, channels) as writer:
        length = 0
        for step in assembler.assemble(
            start_time=start_time.replace(tzinfo=datetime.timezone.utc)
            if start_time
            else None,
            end_time=end_time.replace(tzinfo=datetime.timezone.utc)
            if end_time
            else None,
            scenario=scenario,
            channels=channels,
        ):
            time = int(step["time"])
            statistics_collector.add_all(step)
            writer.write_step(step, time)
            length += 1

    _write_metadata_file(
        output_path,
        _make_metadata(
            mode,
            ["Time"] + channels,
            start_time,
            end_time,
            resolution,
            length,
            statistics_collector.get_all_channel_statistics(),
        ),
    )
