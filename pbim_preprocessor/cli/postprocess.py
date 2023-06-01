from pathlib import Path
from typing import List, Any, Dict, Literal, Optional

import click
import numpy as np

from pbim_preprocessor.post_processor.pbim import (
    DatasetSampler,
)
from pbim_preprocessor.post_processor.sampling import (
    UniformSamplingStrategy,
    WeightedRandomSamplingStrategy,
    HourlySamplingStrategy,
)

StrategyTypes = Literal["uniform", "hourly", "weighted-random"]


def _validate_extra_args(strategy: StrategyTypes, extra_args: Dict[str, Any]):
    def _raise_if_not_present(name: str):
        if name not in extra_args:
            raise click.BadParameter(
                f"Missing argument for strategy {strategy}: {name}"
            )

    match strategy:
        case "uniform":
            _raise_if_not_present("num-windows")
        case "hourly":
            _raise_if_not_present("samples-per-hour")
            _raise_if_not_present("windows-per-sample")
        case "weighted-random":
            _raise_if_not_present("num-windows")
        case _:
            raise click.BadParameter(f"Unknown strategy: {strategy}")


def _make_strategy(
    strategy: StrategyTypes, window_size: int, extra_args: Dict[str, Any]
):
    match strategy:
        case "uniform":
            return UniformSamplingStrategy(
                num_windows=int(extra_args["num-windows"]),
                window_size=window_size,
            )
        case "hourly":
            return HourlySamplingStrategy(
                samples_per_hour=int(extra_args["samples-per-hour"]),
                windows_per_sample=int(extra_args["windows-per-sample"]),
                window_size=window_size,
            )
        case "weighted-random":
            return WeightedRandomSamplingStrategy(
                num_windows=int(extra_args["num-windows"]),
                window_size=window_size,
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
@click.argument("strategy", type=click.Choice(["uniform", "hourly", "weighted-random"]))
@click.option("--window-size", type=int, default=128)
@click.option("--remove-zero-windows", type=bool, default=True)
@click.option("--seed", type=int, default=None)
@click.pass_context
def postprocess(
    ctx,
    input_file: Path,
    output_file: Path,
    strategy: StrategyTypes,
    window_size: int,
    remove_zero_windows: bool,
    seed: Optional[int],
):
    np.random.seed(seed)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    extra_args = _group_extra_args(ctx.args)
    _validate_extra_args(strategy, extra_args)
    strategy = _make_strategy(strategy, window_size, extra_args)
    sampler = DatasetSampler(window_size, remove_zero_windows, strategy)
    sampler.process(input_file, output_file)
