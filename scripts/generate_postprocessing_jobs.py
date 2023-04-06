#!/usr/bin/env python

from pathlib import Path
import jinja2

SAMPLES_PER_HOUR = {
    "mean": 7,
    "interpolate": 7,
    "nosampling": 5,
}

SAMPLE_LENGTH = {"mean": 120, "interpolate": 120, "nosampling": 60}


def load_template(template_path: Path) -> jinja2.Template:
    loader = jinja2.FileSystemLoader(searchpath=template_path.parent)
    template_environment = jinja2.Environment(loader=loader)
    return template_environment.get_template(template_path.name)


def render_template_and_save(template: jinja2.Template, output_path: Path, **kwargs):
    output_path.write_text(template.render(**kwargs))


def get_hourly_strategy_extra_args(aggregation: str) -> str:
    # ~ 5 x 1min at 75Hz
    # ~ 7 x 2min at 25Hz

    return f"--samples-per-hour {SAMPLES_PER_HOUR[aggregation]} --sample-length {SAMPLE_LENGTH[aggregation]}"


def get_extra_args(
    window_size: int, num_samples: int, strategy: str, aggregation: str
) -> str:
    match strategy:
        case "uniform":
            return f"--num-samples {num_samples} --window-size {window_size}"
        case "weighted-random":
            return f"--num-samples {num_samples} --window-size {window_size}"
        case "hourly":
            return get_hourly_strategy_extra_args(aggregation)


if __name__ == "__main__":
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

    NUM_SAMPLES = 2000000
    WINDOW_SIZE = 128
    SCENARIO = "N"

    template = load_template(Path("template/postprocess_pbim_job_template.yml"))
    for strategy in ["uniform", "weighted-random", "hourly"]:
        for aggregation in ["nosampling", "mean", "interpolate"]:
            for filename in FILE_NAMES[SCENARIO]:
                input_path_parameter = (
                    f"/data/PBIM/{SCENARIO}/{aggregation}/{filename}/assembled.dat"
                )
                output_path_parameter = f"/data/PBIM/{SCENARIO}/post-processed/{aggregation}-{strategy}/{filename}/assembled.dat"
                output_path = Path(
                    f"k8s/assemble_jobs/pbim/post-process-jobs/{SCENARIO}/{strategy}/{filename}-{aggregation}-{strategy}.yml"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                render_template_and_save(
                    template,
                    output_path,
                    INPUT_FILE=input_path_parameter,
                    OUTPUT_FILE=output_path_parameter,
                    STRATEGY=strategy,
                    STRATEGY_ARGS=get_extra_args(
                        WINDOW_SIZE, NUM_SAMPLES, strategy, aggregation
                    ),
                    AGGREGATION=aggregation,
                    FILENAME=filename,
                    SEED=42,
                )
