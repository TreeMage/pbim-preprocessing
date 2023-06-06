from pathlib import Path
from typing import List, Any, Dict, Literal, Optional, Tuple

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


def _get_lux_postprocess_params(extra_args: Dict[str, Any]) -> Tuple[Any, ...]:
    sampling_rate = extra_args.get("sampling-rate", None)
    lower_bound_first_frequency = extra_args.get("lower-bound-first-frequency", None)
    upper_bound_first_frequency = extra_args.get("upper-bound-first-frequency")

    lower_bound = extra_args.get("lower-bound", None)
    upper_bound = extra_args.get("upper-bound", None)
    top_k = extra_args.get("top-k", None)
    grace_period = extra_args.get("grace-period", None)

    return (
        sampling_rate,
        lower_bound,
        upper_bound,
        top_k,
        grace_period,
        lower_bound_first_frequency,
        upper_bound_first_frequency,
    )


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
            params = _get_lux_postprocess_params(extra_args)
            existing = [p is not None for p in params]
            if any(existing) and not all(existing):
                _raise_if_not_present("sampling-rate")
                _raise_if_not_present("lower-bound")
                _raise_if_not_present("upper-bound")
                _raise_if_not_present("lower-bound-first-frequency")
                _raise_if_not_present("upper-bound-first-frequency")
                _raise_if_not_present("top-k")
                _raise_if_not_present("grace-period")
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
            params = _get_lux_postprocess_params(extra_args)
            if any([p is None for p in params]):
                filter_config = None
            else:
                (
                    sampling_rate,
                    lower_bound,
                    upper_bound,
                    top_k,
                    grace_period,
                    lower_bound_first_frequency,
                    upper_bound_first_frequency,
                ) = params
                filter_config = LuxDatasetSampler.FilterConfig(
                    sampling_rate=int(sampling_rate),
                    lower_frequency_bound_first_frequency=float(
                        lower_bound_first_frequency
                    ),
                    upper_frequency_bound_first_frequency=float(
                        upper_bound_first_frequency
                    ),
                    lower_frequency_bound=float(lower_bound),
                    upper_frequency_bound=float(upper_bound),
                    top_k=int(top_k),
                    grace_period=int(grace_period),
                )

            return LuxDatasetSampler(
                window_size=window_size,
                sampling_strategy=strategy,
                filter_config=filter_config,
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
