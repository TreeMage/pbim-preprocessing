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
    Z24UndamagedAssembler,
    Z24DamagedAssembler,
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
    "z24-undamaged": [
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
    "z24-damaged": [
        "100V",
        "101V",
        "102V",
        "103V",
        "104V",
        "105V",
        "106V",
        "107V",
        "108V",
        "109V",
        "110V",
        "111V",
        "112V",
        "113V",
        "114V",
        "115V",
        "116V",
        "117V",
        "118V",
        "119V",
        "120V",
        "121V",
        "122V",
        "123V",
        "124V",
        "125V",
        "126V",
        "127V",
        "128V",
        "129V",
        "130V",
        "131V",
        "132V",
        "133V",
        "134V",
        "135V",
        "136V",
        "137V",
        "138V",
        "139V",
        "140V",
        "141V",
        "142V",
        "143V",
        "199L",
        "199T",
        "199V",
        "200T",
        "200V",
        "201T",
        "201V",
        "202T",
        "202V",
        "203L",
        "203T",
        "203V",
        "204L",
        "204T",
        "204V",
        "205T",
        "205V",
        "206T",
        "206V",
        "207T",
        "207V",
        "208L",
        "208T",
        "208V",
        "209L",
        "209T",
        "209V",
        "210T",
        "210V",
        "211T",
        "211V",
        "212T",
        "212V",
        "213L",
        "213T",
        "213V",
        "214L",
        "214T",
        "214V",
        "215T",
        "215V",
        "216T",
        "216V",
        "217T",
        "217V",
        "218L",
        "218T",
        "218V",
        "219L",
        "219T",
        "219V",
        "220T",
        "220V",
        "221T",
        "221V",
        "222T",
        "222V",
        "223L",
        "223T",
        "223V",
        "224L",
        "224T",
        "224V",
        "225T",
        "225V",
        "226T",
        "226V",
        "227T",
        "227V",
        "228L",
        "228T",
        "228V",
        "229L",
        "229T",
        "229V",
        "230T",
        "231T",
        "231V",
        "232T",
        "232V",
        "233L",
        "233V",
        "234L",
        "234T",
        "234V",
        "235T",
        "235V",
        "236T",
        "236V",
        "237T",
        "237V",
        "238L",
        "238T",
        "238V",
        "239L",
        "239T",
        "239V",
        "240T",
        "240V",
        "241T",
        "241V",
        "242T",
        "242V",
        "243L",
        "243T",
        "243V",
        "299V",
        "300V",
        "301V",
        "302V",
        "303V",
        "304V",
        "305V",
        "306V",
        "307V",
        "308V",
        "309V",
        "310V",
        "311V",
        "312V",
        "313V",
        "314V",
        "315V",
        "316V",
        "317V",
        "318V",
        "319V",
        "320V",
        "321V",
        "322V",
        "323V",
        "324V",
        "325V",
        "326V",
        "327V",
        "328V",
        "329V",
        "330V",
        "331V",
        "332V",
        "333V",
        "334V",
        "335V",
        "336V",
        "337V",
        "338V",
        "339V",
        "340V",
        "341V",
        "342V",
        "343V",
        "411L",
        "411T",
        "411V",
        "412L",
        "412T",
        "412V",
        "421L",
        "421T",
        "421V",
        "422L",
        "422T",
        "422V",
        "431L",
        "431T",
        "431V",
        "432L",
        "432T",
        "432V",
        "441L",
        "441T",
        "441V",
        "442L",
        "442T",
        "442V",
        "511L",
        "511T",
        "511V",
        "512L",
        "512T",
        "512V",
        "521L",
        "521T",
        "521V",
        "522L",
        "522T",
        "522V",
        "531L",
        "531T",
        "531V",
        "532L",
        "532T",
        "532V",
        "541L",
        "541T",
        "541V",
        "542L",
        "542T",
        "542V",
        "99V ",
        "R1V ",
        "R2L ",
        "R2T ",
        "R2V ",
        "R3V ",
    ],
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
        case "z24-undamaged":
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
        case "z24-damaged":
            return DatasetMetadata(
                channel_order=["Time"] + channels,
                start_time=start_time.timestamp() if start_time else None,
                end_time=end_time.timestamp() if end_time else None,
                # Channels (including time)
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
    return channels


def _make_assembler(
    mode: str, path: Path, strategy: str, resolution: float
) -> PBimAssembler | GrandStandAssembler | Z24UndamagedAssembler | Z24DamagedAssembler:
    match mode:
        case "pbim":
            return PBimAssembler(path, STRATEGIES[strategy], resolution)
        case "grandstand":
            return GrandStandAssembler(path)
        case "z24-undamaged":
            return Z24UndamagedAssembler(path, STRATEGIES[strategy], resolution)
        case "z24-damaged":
            return Z24DamagedAssembler(path)


@click.command()
@click.argument(
    "mode", type=click.Choice(["pbim", "grandstand", "z24-damaged", "z24-undamaged"])
)
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
@click.option("--z24-mode", type=click.Choice(["avt", "fvt"]))
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
    z24_mode: Optional[str],
):
    LOGGER.set_debug(debug)
    _validate_args(mode, start_time, end_time, resolution, scenario)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    channels = _prepare_channels(mode, list(channel))
    assembler = AssemblerWrapper(
        mode,
        _make_assembler(mode, path, strategy, resolution),
    )

    statistics_collector = StatisticsCollector()
    with _make_writer(output_format, output_path, channels) as writer:
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
            channels=channels,
            mode=z24_mode,
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
            channels,
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
