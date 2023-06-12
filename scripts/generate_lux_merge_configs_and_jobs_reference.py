#!/usr/bin/env python
import json
from pathlib import Path
from typing import Optional

import jinja2


def get_files(scenario: str, aggregation: str, resolution: Optional[str]):
    def block(s: str, anomalous: bool):
        return {
            "relative_path": f"{s}/post-processed/reference/{aggregation}/{resolution}/assembled.dat"
            if resolution is not None
            else f"{s}/post-processed/reference/{aggregation}/assembled.dat",
            "is_anomalous": anomalous,
            "offset": 0.9 if s == "N" else 0,
            "ratio": 0.1 if s == "N" else 1,
            "shuffle": True,
        }

    return [block("N", False), block(scenario, True)]


def load_template(template_path: Path) -> jinja2.Template:
    loader = jinja2.FileSystemLoader(searchpath=template_path.parent)
    template_environment = jinja2.Environment(loader=loader)
    return template_environment.get_template(template_path.name)


def render_template_and_save(template: jinja2.Template, output_path: Path, **kwargs):
    output_path.write_text(template.render(**kwargs))


def get_output_path_config(scenario: str, aggregation: str, resolution: Optional[str]):
    if resolution is not None:
        file_name = f"merge-{aggregation}-{resolution}.json"
    else:
        file_name = f"merge-{aggregation}.json"
    return Path(f"configs/lux/anomalous/{scenario}/reference/{file_name}")


def get_dataset_output_path(scenario: str, aggregation: str, resolution: Optional[str]):
    if resolution is not None:
        return f"/data/LUX/{scenario}/merged/anomalous/reference/{aggregation}/{resolution}/assembled.dat"
    else:
        return f"/data/LUX/{scenario}/merged/anomalous/reference/{aggregation}/assembled.dat"


if __name__ == "__main__":
    job_template = load_template(Path("template/merge_lux_job_template.yml"))
    merge_config_template = load_template(
        Path("template/merge_lux_config_template.json")
    )
    for scenario in ["S1", "S2", "S3", "S4"]:
        for aggregation in ["nosampling", "mean", "interpolate"]:
            for resolution in ["250Hz", "1000Hz", "1500Hz"]:
                agg_res = (
                    f"{aggregation}-{resolution}"
                    if aggregation != "nosampling"
                    else aggregation
                )
                agg_res_path = agg_res.replace("-", "/")
                output_path_job = Path(
                    f"k8s/assemble_jobs/lux/post-process-jobs/{scenario}/merge/reference/merge-{agg_res}.yml"
                )
                output_path_config = get_output_path_config(
                    scenario,
                    aggregation,
                    resolution if aggregation != "nosampling" else None,
                )
                config_path_parameter = Path("/app") / output_path_config
                output_path_config.parent.mkdir(parents=True, exist_ok=True)
                output_path_job.parent.mkdir(parents=True, exist_ok=True)
                render_template_and_save(
                    merge_config_template,
                    output_path_config,
                    MODE="reference",
                    OUTPUT_FILE=get_dataset_output_path(
                        scenario,
                        aggregation,
                        resolution if aggregation != "nosampling" else None,
                    ),
                    AGGREGATION=aggregation,
                    RESOLUTION=resolution,
                    SCENARIO=scenario,
                    FILES=json.dumps(
                        get_files(
                            scenario,
                            aggregation,
                            resolution if aggregation != "nosampling" else None,
                        ),
                        indent=4,
                    ),
                )
                render_template_and_save(
                    job_template,
                    output_path_job,
                    RESOLUTION=resolution.lower(),
                    AGGREGATION=aggregation,
                    SCENARIO=scenario.lower(),
                    CONFIG_PATH=config_path_parameter,
                )
                if aggregation == "nosampling":
                    break
