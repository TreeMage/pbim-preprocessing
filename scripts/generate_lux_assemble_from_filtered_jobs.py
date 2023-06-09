#!/usr/bin/env python
import datetime
from pathlib import Path
import jinja2

SCENARIOS = ["N", "S1", "S2", "S3", "S4"]
FREQUENCIES = [250, 1000, 1500]  # Hz


def load_template(template_path: Path) -> jinja2.Template:
    loader = jinja2.FileSystemLoader(searchpath=template_path.parent)
    template_environment = jinja2.Environment(loader=loader)
    return template_environment.get_template(template_path.name)


def render_template_and_save(template: jinja2.Template, output_path: Path, **kwargs):
    output_path.write_text(template.render(**kwargs))


if __name__ == "__main__":
    template = load_template(Path("template/assemble_job_template_lux.yml"))
    for scenario in SCENARIOS:
        for aggregation in ["mean", "interpolate"]:
            for frequency in FREQUENCIES:
                output_path_job = Path(
                    f"k8s/assemble_jobs/lux/assemble_from_pre_assembled/{scenario}/{aggregation}-{frequency}Hz.yml"
                )
                output_path_job.parent.mkdir(parents=True, exist_ok=True)
                base_path = f"/out/{scenario}/filtered/nosampling/assembled.dat"
                resolution = 1 / frequency
                render_template_and_save(
                    template,
                    output_path_job,
                    BASE_PATH=base_path,
                    OUTPUT_PATH=f"/out/{scenario}/filtered/{aggregation}/{frequency}Hz/assembled.dat",
                    START_TIME=datetime.datetime(
                        1970, 1, 1
                    ),  # Ignored when assembling from pre-assembled
                    END_TIME=datetime.datetime(
                        1970, 1, 1
                    ),  # Ignored when assembling from pre-assembled
                    AGGREGATION=aggregation,
                    SCENARIO=scenario,
                    RESOLUTION=resolution,
                    FREQUENCY=frequency,
                )
