#!/usr/bin/env python

from pathlib import Path
import jinja2


SCENARIOS = ["N", "S1", "S2", "S3", "S4"]


def load_template(template_path: Path) -> jinja2.Template:
    loader = jinja2.FileSystemLoader(searchpath=template_path.parent)
    template_environment = jinja2.Environment(loader=loader)
    return template_environment.get_template(template_path.name)


def render_template_and_save(template: jinja2.Template, output_path: Path, **kwargs):
    output_path.write_text(template.render(**kwargs))


def get_num_samples(
    target_windows: int, window_size: int, strategy: str, aggregation: str
) -> int:
    # windows = samples - window-size + 1 <=> samples = windows + window-size - 1
    match strategy:
        case "uniform":
            return target_windows + window_size - 1
        case _:
            raise NotImplementedError("Only uniform strategy is supported")


def get_extra_args(
    window_size: int, num_windows: int, strategy: str, aggregation: str
) -> str:
    match strategy:
        case "uniform":
            num_samples = get_num_samples(
                num_windows, window_size, strategy, aggregation
            )
            return f"--num-samples {num_samples} --window-size {window_size}"
        case _:
            raise NotImplementedError("Only uniform strategy is supported")


if __name__ == "__main__":
    NUM_WINDOWS = 101000
    WINDOW_SIZE = 128

    template = load_template(Path("template/postprocess_lux_job_template.yml"))
    for scenario in SCENARIOS:
        for aggregation in ["nosampling", "mean", "interpolate"]:
            for frequency in [25, 250, 1000, 1500]:
                if aggregation == "nosampling":
                    input_path_parameter = (
                        f"/data/LUX/{scenario}/assembled/{aggregation}/assembled.dat"
                    )
                    output_path_parameter = f"/data/PBIM/{scenario}/post-processed/{aggregation}/assembled.dat"
                    output_path = Path(
                        f"k8s/assemble_jobs/lux/post-process-jobs/{scenario}/{aggregation}.yml"
                    )
                else:
                    input_path_parameter = f"/data/LUX/{scenario}/assembled/{aggregation}/{frequency}Hz/assembled.dat"
                    output_path_parameter = f"/data/LUX/{scenario}/post-processed/{aggregation}/{frequency}Hz/assembled.dat"
                    output_path = Path(
                        f"k8s/assemble_jobs/lux/post-process-jobs/{scenario}/{aggregation}-{frequency}.yml"
                    )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                render_template_and_save(
                    template,
                    output_path,
                    INPUT_FILE=input_path_parameter,
                    OUTPUT_FILE=output_path_parameter,
                    STRATEGY="uniform",
                    STRATEGY_ARGS=get_extra_args(
                        WINDOW_SIZE, NUM_WINDOWS, "uniform", aggregation
                    ),
                    AGGREGATION=aggregation,
                    SCENARIO=scenario.lower(),
                    FREQUENCY=frequency,
                    SEED=42,
                )
