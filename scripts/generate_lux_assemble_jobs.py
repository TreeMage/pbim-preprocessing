#!/usr/bin/env python

from pathlib import Path
import jinja2

SCENARIOS = ["N", "S1", "S2", "S3", "S4"]
FREQUENCIES = [25, 250, 1000, 1500]  #Hz

START_AND_END_TIMES = {
    "N": ("2014-01-23T00:00:00", "2014-01-31T11:59:59"),
    "S1": ("2014-01-31T12:00:00", "2014-02-04T11:59:59"),
    "S2": ("2014-02-04T12:00:00", "2014-02-06T11:59:59"),
    "S3": ("2014-02-06T12:00:00", "2014-02-11T11:59:59"),
    "S4": ("2014-02-11T12:00:00", "2014-02-20T11:59:59"),
}


def load_template(template_path: Path) -> jinja2.Template:
    loader = jinja2.FileSystemLoader(searchpath=template_path.parent)
    template_environment = jinja2.Environment(loader=loader)
    return template_environment.get_template(template_path.name)


def render_template_and_save(template: jinja2.Template, output_path: Path, **kwargs):
    output_path.write_text(template.render(**kwargs))


if __name__ == "__main__":
    template = load_template(Path("template/assemble_job_template_lux.yml"))
    for scenario in SCENARIOS:
        for aggregation in ["nosampling", "mean", "interpolate"]:
            for frequency in FREQUENCIES:
                if aggregation == "nosampling":
                    output_path = Path(
                        f"k8s/assemble_jobs/lux/assemble/{scenario}/{aggregation}.yml"
                    )
                else:
                    output_path = Path(
                        f"k8s/assemble_jobs/lux/assemble/{scenario}/{aggregation}-{frequency}Hz.yml"
                    )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                start_time, end_time = START_AND_END_TIMES[scenario]
                if aggregation == "nosampling":
                    resolution = 0
                else:
                    resolution = 1 / frequency
                render_template_and_save(
                    template,
                    output_path,
                    START_TIME=start_time,
                    END_TIME=end_time,
                    AGGREGATION=aggregation,
                    SCENARIO=scenario,
                    RESOLUTION=resolution,
                    FREQUENCY=frequency
                )
