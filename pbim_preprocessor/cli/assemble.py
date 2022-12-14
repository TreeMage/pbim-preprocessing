import datetime
from pathlib import Path

import click

from pbim_preprocessor.assembler import Assembler
from pbim_preprocessor.parser import POST_PROCESSABLE_CHANNELS
from pbim_preprocessor.sampling import (
    MeanSamplingStrategy,
    LinearInterpolationSamplingStrategy,
)
from pbim_preprocessor.writer import CsvWriter

STRATEGIES = {
    "mean": MeanSamplingStrategy(),
    "interpolate": LinearInterpolationSamplingStrategy(),
}

CHANNELS = POST_PROCESSABLE_CHANNELS


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument(
    "output-path", type=click.Path(writable=True, dir_okay=False, path_type=Path)
)
@click.argument("start-time", type=click.DateTime())
@click.argument("end-time", type=click.DateTime())
@click.argument("resolution", type=click.INT)
@click.option("--strategy", default="mean", type=click.Choice(list(STRATEGIES.keys())))
def assemble(
    path: Path,
    output_path: Path,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    resolution: int,
    strategy: str,
):
    output_path.parent.mkdir(exist_ok=True, parents=True)
    assembler = Assembler(STRATEGIES[strategy], resolution)
    with CsvWriter(output_path, CHANNELS + ["time"]) as writer:
        for step in assembler.assemble(
            path, start_time=start_time, end_time=end_time, channels=CHANNELS
        ):
            writer.write_step(step)
