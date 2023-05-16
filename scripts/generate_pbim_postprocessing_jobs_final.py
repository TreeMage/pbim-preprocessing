#!/usr/bin/env python
import math
from pathlib import Path
import jinja2


FILE_NAMES = {
    # Normal
    "N": [
        "april-week-01",
        "january-week-01",
        "june-week-01",
        "may-week-01",
        "february-week-01",
        "july-week-01",
        "march-week-01",
    ],
    # Damaged
    "S1": [
        "july-week-02",
        "august-week-01",
    ],
    "S2": [
        "july-week-02",
        "august-week-01",
    ],
    "S3": [
        "july-week-02",
        "august-week-01",
    ],
}

SAMPLES_PER_HOUR = 4
SAMPLE_RESOLUTION = {"mean": 25, "interpolate": 25, "nosampling": 75}
DATASET_LENGTH_IN_HOURS = 4 * 24


def load_template(template_path: Path) -> jinja2.Template:
    loader = jinja2.FileSystemLoader(searchpath=template_path.parent)
    template_environment = jinja2.Environment(loader=loader)
    return template_environment.get_template(template_path.name)


def render_template_and_save(template: jinja2.Template, output_path: Path, **kwargs):
    output_path.write_text(template.render(**kwargs))


def get_extra_args(window_size: int, num_windows: int) -> str:
    return f"--num-samples {num_windows} --window-size {window_size}"


if __name__ == "__main__":
    NUM_WINDOWS_N = 2915962
    NUM_WINDOWS_S = 1133985
    WINDOW_SIZE = 256

    template = load_template(Path("template/postprocess_pbim_job_template.yml"))
    # Normal data
    scenario = "N"
    for filename in FILE_NAMES[scenario]:
        input_path_parameter = (
            f"/data/PBIM/{scenario}/assembled/nosampling/{filename}/assembled.dat"
        )
        output_path_parameter = (
            f"/data/PBIM/{scenario}/post-processed/final/{filename}/assembled.dat"
        )
        output_path = Path(
            f"k8s/assemble_jobs/pbim/post-process-jobs/final/{scenario}/{filename}.yml"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        render_template_and_save(
            template,
            output_path,
            INPUT_FILE=input_path_parameter,
            OUTPUT_FILE=output_path_parameter,
            STRATEGY="uniform",
            STRATEGY_ARGS=get_extra_args(WINDOW_SIZE, NUM_WINDOWS_N),
            AGGREGATION="nosampling",
            FILENAME=filename,
            SCENARIO=scenario.lower(),
            SEED=42,
        )
    # Damaged data
    for scenario in ["S1", "S2", "S3"]:
        for filename in FILE_NAMES[scenario]:
            input_path_parameter = (
                f"/data/PBIM/{scenario}/assembled/nosampling/{filename}/assembled.dat"
            )
            output_path_parameter = (
                f"/data/PBIM/{scenario}/post-processed/final/{filename}/assembled.dat"
            )
            output_path = Path(
                f"k8s/assemble_jobs/pbim/post-process-jobs/final/{scenario}/{filename}.yml"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            render_template_and_save(
                template,
                output_path,
                INPUT_FILE=input_path_parameter,
                OUTPUT_FILE=output_path_parameter,
                STRATEGY="uniform",
                STRATEGY_ARGS=get_extra_args(WINDOW_SIZE, NUM_WINDOWS_S),
                AGGREGATION="nosampling",
                FILENAME=filename,
                SCENARIO=scenario.lower(),
                SEED=42,
            )
