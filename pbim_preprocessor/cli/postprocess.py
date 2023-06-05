from pathlib import Path
from typing import List, Any, Dict, Literal, Optional

import click
import numpy as np

from pbim_preprocessor.post_processor.dataset_post_processor import (
    PBimDatasetSampler,
    LuxDatasetSampler,
)
from pbim_preprocessor.post_processor.sampling import (
    UniformSamplingStrategy,
    WeightedRandomSamplingStrategy,
    HourlySamplingStrategy,
    NoopSamplingStrategy,
    DatasetSamplingStrategy,
)

StrategyTypes = Literal["uniform", "hourly", "weighted-random"]
Modes = Literal["pbim", "lux"]


def _validate_extra_args(
    mode: Modes, strategy: StrategyTypes, extra_args: Dict[str, Any]
):
    def _raise_if_not_present(name: str):
        if name not in extra_args:
            raise click.BadParameter(
                f"Missing argument for strategy {strategy} and mode {mode}: {name}"
            )

    match strategy:
        case "uniform":
            _raise_if_not_present("num-windows")
        case "hourly":
            _raise_if_not_present("samples-per-hour")
            _raise_if_not_present("windows-per-sample")
        case "weighted-random":
            _raise_if_not_present("num-windows")
        case "noop":
            pass
        case _:
            raise click.BadParameter(f"Unknown strategy: {strategy}")

    match mode:
        case "pbim":
            pass
        case "lux":
            _raise_if_not_present("sampling-rate")
            _raise_if_not_present("lower-bound")
            _raise_if_not_present("upper-bound")
            _raise_if_not_present("lower-bound-first-frequency")
            _raise_if_not_present("upper-bound-first-frequency")
            _raise_if_not_present("top-k")
        case _:
            raise click.BadParameter(f"Unknown mode: {mode}")


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
        case "noop":
            return NoopSamplingStrategy()


def _make_sampler(
    mode: str,
    window_size: int,
    strategy: DatasetSamplingStrategy,
    extra_args: Dict[str, Any],
):
    match mode:
        case "pbim":
            return PBimDatasetSampler(
                window_size=window_size,
                sampling_strategy=strategy,
            )
        case "lux":
            sampling_rate = int(extra_args["sampling-rate"])
            lower_bound_first_frequency = float(
                extra_args["lower-bound-first-frequency"]
            )
            upper_bound_first_frequency = float(
                extra_args["upper-bound-first-frequency"]
            )
            lower_bound = float(extra_args["lower-bound"])
            upper_bound = float(extra_args["upper-bound"])
            top_k = int(extra_args["top-k"])
            return LuxDatasetSampler(
                window_size=window_size,
                sampling_strategy=strategy,
                sampling_rate=sampling_rate,
                lower_frequency_bound_first_frequency=lower_bound_first_frequency,
                upper_frequency_bound_first_frequency=upper_bound_first_frequency,
                lower_frequency_bound=lower_bound,
                upper_frequency_bound=upper_bound,
                top_k=top_k,
            )


def _group_extra_args(extra_args: List[Any]) -> Dict[str, Any]:
    return {
        name.replace("--", ""): value
        for name, value in zip(extra_args[::2], extra_args[1::2])
    }


@click.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
@click.argument("mode", type=click.Choice(["pbim", "lux"]))
@click.argument(
    "input-file", type=click.Path(exists=True, path_type=Path, dir_okay=False)
)
@click.argument(
    "output-file", type=click.Path(exists=False, path_type=Path, dir_okay=False)
)
@click.argument(
    "strategy", type=click.Choice(["uniform", "hourly", "weighted-random", "noop"])
)
@click.option("--window-size", type=int, default=128)
@click.option("--seed", type=int, default=None)
@click.pass_context
def postprocess(
    ctx,
    mode: Modes,
    input_file: Path,
    output_file: Path,
    strategy: StrategyTypes,
    window_size: int,
    seed: Optional[int],
):
    np.random.seed(seed)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    extra_args = _group_extra_args(ctx.args)
    _validate_extra_args(mode, strategy, extra_args)
    strategy = _make_strategy(strategy, window_size, extra_args)
    sampler = _make_sampler(mode, window_size, strategy, extra_args)
    sampler.process(input_file, output_file)
