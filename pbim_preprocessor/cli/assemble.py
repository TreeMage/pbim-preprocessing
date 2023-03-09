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
    Z24Assembler,
    MergeChannelsConfig,
)
from pbim_preprocessor.index import _write_index
from pbim_preprocessor.model import EOF
from pbim_preprocessor.parser import POST_PROCESSABLE_CHANNELS
from pbim_preprocessor.sampling import (
    MeanSamplingStrategy,
    LinearInterpolationSamplingStrategy,
)
from pbim_preprocessor.statistic import StatisticsCollector, ChannelStatistics
from pbim_preprocessor.utils import LOGGER
from pbim_preprocessor.writer import CsvWriter, BinaryWriter

TIME_BYTE_SIZE = 8

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
    "grandstand": [f"Joint {i}" for i in range(1, 31)],
    "z24": [
        "WS",
        "WD",
        "AT",
        "R",
        "H",
        "TE",
        "ADU",
        "ADK",
        "TSPU1",
        "TSPU2",
        "TSPU3",
        "TSAU1",
        "TSAU2",
        "TSAU3",
        "TSPK1",
        "TSPK2",
        "TSPK3",
        "TSAK1",
        "TSAK2",
        "TSAK3",
        "TBC1",
        "TBC2",
        "TSWS1",
        "TSWN1",
        "TWS1",
        "TWC1",
        "TWN1",
        "TP1",
        "TDT1",
        "TDS1",
        "TS1",
        "TSWS2",
        "TSWN2",
        "TWS2",
        "TWC2",
        "TWN2",
        "TP2",
        "TDT2",
        "TDS2",
        "TS2",
        "TWS3",
        "TWN3",
        "TWC3",
        "TP3",
        "TDT3",
        "TS3",
        "03",
        "05",
        "06",
        "07",
        "10",
        "12",
        "14",
        "16",
    ],
}

MERGE_CONFIGS = {
    "z24": [MergeChannelsConfig(["TBC1", "TBC2"], "TBC")],
    "pbim": [],
    "grandstand": [],
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
    time_byte_size: int


def _write_metadata_file(path: Path, metadata: DatasetMetadata):
    metadata_path = path.parent / f"{path.stem}.metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(
            metadata.to_dict(),
            f,
            indent=4,
        )


def _make_writer(write_type: str, path: Path, headers: List[str], **kwargs):
    match write_type:
        case "csv":
            delimiter = kwargs.get("delimiter", ",")
            return CsvWriter(path, headers, delimiter)
        case "binary":
            return BinaryWriter(path, headers, TIME_BYTE_SIZE)


def _make_metadata(
    mode: str,
    channels: List[str],
    start_time: Optional[datetime.datetime],
    end_time: Optional[datetime.datetime],
    resolution: Optional[int],
    length: int,
    statistics: Dict[str, ChannelStatistics],
    time_byte_size: int,
) -> DatasetMetadata:
    match mode:
        case "pbim":
            return DatasetMetadata(
                channel_order=["Time"] + channels,
                start_time=int(start_time.timestamp()) if start_time else None,
                end_time=int(end_time.timestamp()) if end_time else None,
                # Time + Channels
                measurement_size_in_bytes=time_byte_size + len(channels) * 4,
                resolution=resolution,
                length=length,
                statistics=statistics,
                time_byte_size=time_byte_size,
            )
        case "grandstand":
            return DatasetMetadata(
                channel_order=["Time"] + channels,
                start_time=None,
                end_time=None,
                # Channels (including time)
                measurement_size_in_bytes=(len(channels) - 1) * 4 + time_byte_size,
                resolution=None,
                length=length,
                statistics=statistics,
                time_byte_size=time_byte_size,
            )
        case "z24":
            return DatasetMetadata(
                channel_order=["Time"] + channels,
                start_time=start_time.timestamp() if start_time else None,
                end_time=end_time.timestamp() if end_time else None,
                measurement_size_in_bytes=len(channels) * 4 + time_byte_size,
                resolution=resolution,
                length=length,
                statistics=statistics,
                time_byte_size=time_byte_size,
            )


def _validate_args(
    mode: str,
    start_time: Optional[datetime.datetime],
    end_time: Optional[datetime.datetime],
    resolution: Optional[float],
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
        case "z24":
            if resolution is None:
                _raise("resolution")


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
        case "z24":
            if "all" in channels:
                channels = CHANNELS["z24"]
            if "relevant" in channels:
                channels = [
                    "03",
                    "05",
                    "06",
                    "07",
                    "10",
                    "12",
                    "14",
                    "16",
                    "TBC1",
                    "TBC2",
                ]
    return channels


def _make_assembler(
    mode: str, path: Path, strategy: str, resolution: float
) -> PBimAssembler | GrandStandAssembler | Z24Assembler:
    match mode:
        case "pbim":
            return PBimAssembler(path, STRATEGIES[strategy], resolution)
        case "grandstand":
            return GrandStandAssembler(path)
        case "z24":
            return Z24Assembler(
                path, STRATEGIES[strategy], resolution, MERGE_CONFIGS["z24"]
            )


def _compute_actual_channels(
    channels: List[str], merge_configs: List[MergeChannelsConfig]
):
    actual_channels = channels.copy()
    for merge_config in merge_configs:
        if all(channel in actual_channels for channel in merge_config.channels):
            if merge_config.remove_original:
                for channel in merge_config.channels:
                    actual_channels.remove(channel)
            actual_channels.append(merge_config.name)
        else:
            raise ValueError(
                f"Supposed to merge channels {merge_config.channels} but at least one of them is missing."
            )
    return actual_channels


@click.command()
@click.argument("mode", type=click.Choice(["pbim", "grandstand", "z24"]))
@click.option("--scenario", type=click.STRING)
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.argument(
    "output-path", type=click.Path(writable=True, dir_okay=False, path_type=Path)
)
@click.option("--start-time", default=None, type=click.DateTime())
@click.option("--end-time", default=None, type=click.DateTime())
@click.option("--resolution", default=None, type=click.FLOAT)
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
    resolution: Optional[float],
    strategy: Optional[str],
    output_format: str,
    channel: List[str],
    debug: bool,
):
    LOGGER.set_debug(debug)
    _validate_args(mode, start_time, end_time, resolution, scenario)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    channels_in = _prepare_channels(mode, list(channel))
    channels_out = _compute_actual_channels(channels_in, MERGE_CONFIGS[mode])

    assembler = AssemblerWrapper(
        mode,
        _make_assembler(mode, path, strategy, resolution),
    )

    statistics_collector = StatisticsCollector()
    with _make_writer(output_format, output_path, channels_out) as writer:
        length = 0
        index = []
        for step in assembler.assemble(
            start_time=start_time.replace(tzinfo=datetime.timezone.utc)
            if start_time
            else None,
            end_time=end_time.replace(tzinfo=datetime.timezone.utc)
            if end_time
            else None,
            scenario=scenario,
            channels=channels_in,
        ):
            if isinstance(step, dict):
                time = int(step["time"])
                statistics_collector.add_all(step)
                writer.write_step(step, time)
                length += 1
            elif isinstance(step, EOF):
                index += [length]

    _write_metadata_file(
        output_path,
        _make_metadata(
            mode,
            channels_out,
            start_time,
            end_time,
            resolution,
            length,
            statistics_collector.get_all_channel_statistics(),
            TIME_BYTE_SIZE,
        ),
    )
    _write_index(
        [length] if not index else index,
        False,
        output_path.parent / f"{output_path.stem}.index.json",
    )
