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
            return target_windows
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
            return f"--num-windows {num_samples} --window-size {window_size}"
        case _:
            raise NotImplementedError("Only uniform strategy is supported")


if __name__ == "__main__":
    NUM_WINDOWS_N = 9341378
    NUM_WINDOWS_S = 934137
    WINDOW_SIZE = 256

    template = load_template(Path("template/postprocess_lux_job_template.yml"))
    for scenario in SCENARIOS:
        num_windows = NUM_WINDOWS_N if scenario == "N" else NUM_WINDOWS_S
        input_path_parameter = f"/data/LUX/{scenario}/filtered/nosampling/assembled.dat"
        output_path_parameter = (
            f"/data/LUX/{scenario}/post-processed/sweep/nosampling/assembled.dat"
        )
        output_path = Path(
            f"k8s/assemble_jobs/lux/post-process-jobs/sweep/{scenario}/nosampling.yml"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        render_template_and_save(
            template,
            output_path,
            INPUT_FILE=input_path_parameter,
            OUTPUT_FILE=output_path_parameter,
            STRATEGY="uniform",
            STRATEGY_ARGS=get_extra_args(
                WINDOW_SIZE, num_windows, "uniform", "nosampling"
            ),
            AGGREGATION="nosampling",
            SCENARIO=scenario.lower(),
            FREQUENCY=0,
            SEED=42,
        )
