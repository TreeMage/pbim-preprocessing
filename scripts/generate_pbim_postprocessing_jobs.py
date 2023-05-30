#!/usr/bin/env python
from pathlib import Path
import jinja2

# 100k
# EMPIRICAL_SCALING_FACTOR_MEAN_AND_INTERPOLATE = (
#    0.035  # was 0.06(++) 0.01(+) 0.008 (--) 0.0095 (--) 0.05 (++)
# )
# EMPIRICAL_SCALING_FACTOR_NOSAMPLING = 0.05  # was 0.09 (++)
# 200k
# EMPIRICAL_SCALING_FACTOR_MEAN_AND_INTERPOLATE = 0.026  # was 0.035 0.025
# EMPIRICAL_SCALING_FACTOR_NOSAMPLING = 0.035  # was 0.05 (++) 0.04 (+)

CORRECTION_FACTOR_HOURLY_MEAN_AND_INTERPOLATE = (
    1.314  # was 1.315 (+) 1.305 (-) 1.310 (-) 1.313 (-)
)
CORRECTION_FACTOR_HOURLY_NOSAMPLING = 1.13  # was 1.315 (+) 1.2 (+) 1.16 (+)

EMPIRICAL_SCALING_FACTOR_MEAN_AND_INTERPOLATE_WEEK_1 = (
    4.52  # 1 (--) # was 4 (-) 4.3 (-) 4.35 (-) 4.45 (-)
)
EMPIRICAL_SCALING_FACTOR_NOSAMPLING_WEEK_1 = 5.5  # 1 (--) # 4 (-) 4.8 (-)

EMPIRICAL_SCALING_FACTOR_MEAN_AND_INTERPOLATE_WEEK_2 = 1
EMPIRICAL_SCALING_FACTOR_NOSAMPLING_WEEK_2 = 1

EMPIRICAL_SCALING_FACTOR_MEAN_AND_INTERPOLATE_WEEK_3 = 1
EMPIRICAL_SCALING_FACTOR_NOSAMPLING_WEEK_3 = 1

EMPIRICAL_SCALING_FACTOR_MEAN_AND_INTERPOLATE_WEEK_4 = 1
EMPIRICAL_SCALING_FACTOR_NOSAMPLING_WEEK_4 = 1

WINDOW_SIZE = 256
MONTHS_UNDAMAGED = [
    "april",
    #    "january",
    "june",
    "may",
    "february",
    "march",
]

MONTHS_DAMAGED = [
    "july",
    "august",
]

SAMPLES_PER_HOUR = 4
SAMPLE_RESOLUTION = {"mean": 25, "interpolate": 25, "nosampling": 75}
DATASET_LENGTH_IN_HOURS = 4 * 24


def load_template(template_path: Path) -> jinja2.Template:
    loader = jinja2.FileSystemLoader(searchpath=template_path.parent)
    template_environment = jinja2.Environment(loader=loader)
    return template_environment.get_template(template_path.name)


def render_template_and_save(template: jinja2.Template, output_path: Path, **kwargs):
    output_path.write_text(template.render(**kwargs))


def get_hourly_strategy_extra_args(
    target_windows: int, window_size: int, dataset_length_in_hours, aggregation: str
) -> str:
    windows_per_sample = target_windows / (SAMPLES_PER_HOUR * dataset_length_in_hours)
    correction_factor = (
        CORRECTION_FACTOR_HOURLY_NOSAMPLING
        if aggregation == "nosampling"
        else CORRECTION_FACTOR_HOURLY_MEAN_AND_INTERPOLATE
    )
    samples_per_hourly_sample = round(windows_per_sample * correction_factor)
    # correction = 1 if aggregation == "nosampling" else 6  # 100k samples
    # correction = 1 if aggregation == "nosampling" else 9  # 200k samples
    # correction = 2 if aggregation == "nosampling" else 14  # 666k samples
    return f"--samples-per-hour {SAMPLES_PER_HOUR} --sample-length {samples_per_hourly_sample} --window-size {window_size}"


def get_scaling_factor(week: int, aggregation: str) -> float:
    match week:
        case 1:
            return (
                EMPIRICAL_SCALING_FACTOR_NOSAMPLING_WEEK_1
                if aggregation == "nosampling"
                else EMPIRICAL_SCALING_FACTOR_MEAN_AND_INTERPOLATE_WEEK_1
            )
        case 2:
            return (
                EMPIRICAL_SCALING_FACTOR_NOSAMPLING_WEEK_2
                if aggregation == "nosampling"
                else EMPIRICAL_SCALING_FACTOR_MEAN_AND_INTERPOLATE_WEEK_2
            )
        case 3:
            return (
                EMPIRICAL_SCALING_FACTOR_NOSAMPLING_WEEK_3
                if aggregation == "nosampling"
                else EMPIRICAL_SCALING_FACTOR_MEAN_AND_INTERPOLATE_WEEK_3
            )
        case 4:
            return (
                EMPIRICAL_SCALING_FACTOR_NOSAMPLING_WEEK_4
                if aggregation == "nosampling"
                else EMPIRICAL_SCALING_FACTOR_MEAN_AND_INTERPOLATE_WEEK_4
            )
        case _:
            raise ValueError(f"Invalid week: {week}")


def get_num_samples(
    target_windows: int, window_size: int, strategy: str, aggregation: str, week: int
) -> int:
    # windows = samples - window-size + 1 <=> samples = windows + window-size - 1
    match strategy:
        case "uniform":
            return target_windows
        case "weighted-random":
            scaling_factor = get_scaling_factor(week, aggregation)
            return int(target_windows * scaling_factor)
        case "hourly":
            raise NotImplementedError("Use get_hourly_strategy_extra_args instead")


def get_extra_args(
    window_size: int, num_windows: int, strategy: str, aggregation: str, week: int
) -> str:
    match strategy:
        case "uniform":
            num_samples = get_num_samples(
                num_windows, window_size, strategy, aggregation, week
            )
            return f"--num-samples {num_samples} --window-size {window_size}"
        case "weighted-random":
            num_samples = get_num_samples(
                num_windows, window_size, strategy, aggregation, week
            )
            return f"--num-samples {num_samples} --window-size {window_size}"
        case "hourly":
            return get_hourly_strategy_extra_args(
                num_windows, window_size, DATASET_LENGTH_IN_HOURS, aggregation
            )


def render_for_all_strategies_and_aggregations(
    scenario: str, num_windows: int, month: str, week: int, template: jinja2.Template
):
    filename = f"{month}-week-{week:02d}"
    for strategy in ["uniform", "weighted-random", "hourly"]:
        for aggregation in ["nosampling", "mean", "interpolate"]:
            input_path_parameter = f"/data/PBIM/{scenario}/assembled/{aggregation}/{filename}/assembled.dat"
            output_path_parameter = f"/data/PBIM/{scenario}/post-processed/{aggregation}-{strategy}/{filename}/assembled.dat"
            output_path = Path(
                f"k8s/assemble_jobs/pbim/post-process-jobs/{scenario}/{strategy}/{week:02d}/{filename}-{aggregation}-{strategy}.yml"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            render_template_and_save(
                template,
                output_path,
                INPUT_FILE=input_path_parameter,
                OUTPUT_FILE=output_path_parameter,
                STRATEGY=strategy,
                STRATEGY_ARGS=get_extra_args(
                    WINDOW_SIZE, num_windows, strategy, aggregation, week
                ),
                AGGREGATION=aggregation,
                FILENAME=filename,
                SCENARIO=scenario.lower(),
                SEED=42,
            )


def main():
    NUM_WINDOWS_TRAINING = 666666
    NUM_WINDOWS_VALIDATION = 41666
    NUM_WINDOWS_THRESHOLD = 41666
    NUM_WINDOWS_TEST_UNDAMGED = 83333
    NUM_WINDOWS_TEST_DAMAGED = 250000

    template = load_template(Path("template/postprocess_pbim_job_template.yml"))
    # Training data
    for month in MONTHS_UNDAMAGED:
        render_for_all_strategies_and_aggregations(
            "N", NUM_WINDOWS_TRAINING, month, 1, template
        )
    # Validation data
    for month in MONTHS_UNDAMAGED:
        render_for_all_strategies_and_aggregations(
            "N", NUM_WINDOWS_VALIDATION, month, 2, template
        )

    # Threshold data
    for month in MONTHS_UNDAMAGED:
        render_for_all_strategies_and_aggregations(
            "N", NUM_WINDOWS_THRESHOLD, month, 3, template
        )

    # Test data non-damaged
    for month in MONTHS_UNDAMAGED:
        render_for_all_strategies_and_aggregations(
            "N", NUM_WINDOWS_TEST_UNDAMGED, month, 4, template
        )

    # Test damaged
    for scenario in ["S1", "S2", "S3"]:
        render_for_all_strategies_and_aggregations(
            scenario, NUM_WINDOWS_TEST_DAMAGED, "july", 2, template
        )
        render_for_all_strategies_and_aggregations(
            scenario, NUM_WINDOWS_TEST_DAMAGED, "august", 1, template
        )


if __name__ == "__main__":
    main()
