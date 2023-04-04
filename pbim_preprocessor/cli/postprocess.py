from pathlib import Path
from typing import List, Any, Dict, Literal

import click

from pbim_preprocessor.post_processor.pbim import (
    UniformSamplingStrategy,
    HourlySamplingStrategy,
    PBimSampler,
    MinutelySamplingStrategy,
    WeightedRandomSamplingStrategy,
)

StrategyTypes = Literal["uniform", "hourly", "minutely"]


def _validate_extra_args(strategy: StrategyTypes, extra_args: Dict[str, Any]):
    def _raise_if_not_present(name: str):
        if name not in extra_args:
            raise click.BadParameter(
                f"Missing argument for strategy {strategy}: {name}"
            )

    match strategy:
        case "uniform":
            _raise_if_not_present("num-samples")
        case "hourly":
            _raise_if_not_present("samples-per-hour")
        case "minutely":
            _raise_if_not_present("samples-per-minute")


def _make_strategy(strategy: StrategyTypes, extra_args: Dict[str, Any]):
    match strategy:
        case "uniform":
            return UniformSamplingStrategy(
                num_samples=int(extra_args["num-samples"]),
                window_size=int(extra_args["window-size"]),
            )
        case "hourly":
            return HourlySamplingStrategy(
                samples_per_hour=int(extra_args["samples-per-hour"]),
                sample_length_in_seconds=int(extra_args["sample-length"]),
            )
        case "minutely":
            return MinutelySamplingStrategy(
                samples_per_minute=int(extra_args["samples-per-minute"]),
                sample_length_in_seconds=int(extra_args["sample-length"]),
            )
        case "weighted-random":
            return WeightedRandomSamplingStrategy(
                num_samples=int(extra_args["num-samples"]),
                window_size=int(extra_args["window-size"]),
            )


def _group_extra_args(extra_args: List[Any]) -> Dict[str, Any]:
    return {
        name.replace("--", ""): value
        for name, value in zip(extra_args[::2], extra_args[1::2])
    }


@click.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
@click.argument(
    "input-file", type=click.Path(exists=True, path_type=Path, dir_okay=False)
)
@click.argument(
    "output-file", type=click.Path(exists=False, path_type=Path, dir_okay=False)
)
@click.argument("strategy", type=click.Choice(["uniform", "hourly", "minutely"]))
@click.option("--window-size", type=int, default=128)
@click.option("--remove-zero-windows", type=bool, default=True, is_flag=True)
@click.pass_context
def postprocess(
    ctx,
    input_file: Path,
    output_file: Path,
    strategy: StrategyTypes,
    window_size: int,
    remove_zero_windows: bool,
):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    extra_args = _group_extra_args(ctx.args)
    _validate_extra_args(strategy, extra_args)
    strategy = _make_strategy(strategy, extra_args)
    sampler = PBimSampler(window_size, remove_zero_windows, strategy)
    sampler.process(input_file, output_file)
