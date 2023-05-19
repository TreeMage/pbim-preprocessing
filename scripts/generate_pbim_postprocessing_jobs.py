#!/usr/bin/env python
import math
from pathlib import Path
import jinja2

# 100k
# EMPIRICAL_SCALING_FACTOR_MEAN_AND_INTERPOLATE = (
#    0.035  # was 0.06(++) 0.01(+) 0.008 (--) 0.0095 (--) 0.05 (++)
# )
# EMPIRICAL_SCALING_FACTOR_NOSAMPLING = 0.05  # was 0.09 (++)
# 200k
#EMPIRICAL_SCALING_FACTOR_MEAN_AND_INTERPOLATE = 0.026  # was 0.035 0.025
#EMPIRICAL_SCALING_FACTOR_NOSAMPLING = 0.035  # was 0.05 (++) 0.04 (+)


EMPIRICAL_SCALING_FACTOR_MEAN_AND_INTERPOLATE = 1#0.018
EMPIRICAL_SCALING_FACTOR_NOSAMPLING = 1#0.017

WINDOW_SIZE = 256
FILE_NAME_TEMPLATES_UNDAMAGED = [
    "april-week",
    "january-week",
    "june-week",
    "may-week",
    "february-week",
    "march-week",
]

FILE_NAME_TEMPLATES_DAMAGED = [
    "july-week",
    "august-week",
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
    samples_per_hourly_sample = math.ceil(windows_per_sample + window_size - 1)
    sample_length = math.ceil(
        samples_per_hourly_sample / SAMPLE_RESOLUTION[aggregation]
    )
    # correction = 1 if aggregation == "nosampling" else 6  # 100k samples
    #correction = 1 if aggregation == "nosampling" else 9  # 200k samples
    correction = 1 if aggregation == "nosampling" else 9  # 666k samples
    return f"--samples-per-hour {SAMPLES_PER_HOUR + correction} --sample-length {sample_length}"


def get_num_samples(
    target_windows: int, window_size: int, strategy: str, aggregation: str
) -> int:
    # windows = samples - window-size + 1 <=> samples = windows + window-size - 1
    match strategy:
        case "uniform":
            return target_windows
        case "weighted-random":
            scaling_factor = (
                EMPIRICAL_SCALING_FACTOR_NOSAMPLING
                if aggregation == "nosampling"
                else EMPIRICAL_SCALING_FACTOR_MEAN_AND_INTERPOLATE
            )
            return int(target_windows * scaling_factor)
        case "hourly":
            raise NotImplementedError("Use get_hourly_strategy_extra_args instead")


def get_extra_args(
    window_size: int, num_windows: int, strategy: str, aggregation: str
) -> str:
    match strategy:
        case "uniform":
            num_samples = get_num_samples(
                num_windows, window_size, strategy, aggregation
            )
            return f"--num-samples {num_samples} --window-size {window_size}"
        case "weighted-random":
            num_samples = get_num_samples(
                num_windows, window_size, strategy, aggregation
            )
            return f"--num-samples {num_samples} --window-size {window_size}"
        case "hourly":
            return get_hourly_strategy_extra_args(
                num_windows, window_size, DATASET_LENGTH_IN_HOURS, aggregation
            )


def render_for_all_strategies_and_aggregations(
    scenario: str, num_windows: int, filename: str, template: jinja2.Template
):
    for strategy in ["uniform", "weighted-random", "hourly"]:
        for aggregation in ["nosampling", "mean", "interpolate"]:
            input_path_parameter = f"/data/PBIM/{scenario}/assembled/{aggregation}/{filename}/assembled.dat"
            output_path_parameter = f"/data/PBIM/{scenario}/post-processed/{aggregation}-{strategy}/{filename}/assembled.dat"
            output_path = Path(
                f"k8s/assemble_jobs/pbim/post-process-jobs/{scenario}/{strategy}/{filename}-{aggregation}-{strategy}.yml"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            render_template_and_save(
                template,
                output_path,
                INPUT_FILE=input_path_parameter,
                OUTPUT_FILE=output_path_parameter,
                STRATEGY=strategy,
                STRATEGY_ARGS=get_extra_args(
                    WINDOW_SIZE, num_windows, strategy, aggregation
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
    NUM_WINDOWS_TEST = 83333

    template = load_template(Path("template/postprocess_pbim_job_template.yml"))
    # Training data
    for file_name_template in FILE_NAME_TEMPLATES_UNDAMAGED:
        file_name = f"{file_name_template}-01"
        render_for_all_strategies_and_aggregations(
            "N", NUM_WINDOWS_TRAINING, file_name, template
        )
    # Validation data
    for file_name_template in FILE_NAME_TEMPLATES_UNDAMAGED:
        file_name = f"{file_name_template}-02"
        render_for_all_strategies_and_aggregations(
            "N", NUM_WINDOWS_VALIDATION, file_name, template
        )

    # Threshold data
    for file_name_template in FILE_NAME_TEMPLATES_UNDAMAGED:
        file_name = f"{file_name_template}-03"
        render_for_all_strategies_and_aggregations(
            "N", NUM_WINDOWS_THRESHOLD, file_name, template
        )

    # Test
    for file_name_template in FILE_NAME_TEMPLATES_DAMAGED:
        file_name = f"{file_name_template}-04"
        for scenario in ["S1", "S2", "S3"]:
            render_for_all_strategies_and_aggregations(
                scenario, NUM_WINDOWS_TEST, file_name, template
            )


if __name__ == "__main__":
    main()
