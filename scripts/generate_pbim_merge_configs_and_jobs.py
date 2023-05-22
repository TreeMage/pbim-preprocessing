#!/usr/bin/env python
import json
from pathlib import Path
from typing import Optional

import jinja2

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


def get_files(
    scenario: str, strategy: str, aggregation: str, mode: Optional[str] = None
):
    def block(s: str, name: str, anomalous: bool):
        return {
            "relative_path": f"{s}/post-processed/{aggregation}-{strategy}/{name}/assembled.dat",
            "is_anomalous": anomalous,
            "include_in_statistics": scenario == "N",
        }

    match scenario:
        case "N":
            match mode:
                case "train":
                    week = 1
                case "val":
                    week = 2
                case "threshold":
                    week = 3
                case _:
                    raise ValueError(f"Unknown mode: {mode}")
            return [
                block("N", f"{file_name_template}-{week:02d}", False)
                for file_name_template in FILE_NAME_TEMPLATES_UNDAMAGED
            ]
        case _:
            undamaged_blocks = [
                block("N", f"{file_name_template}-04", False)
                for file_name_template in FILE_NAME_TEMPLATES_UNDAMAGED
            ]
            damaged_blocks = [
                block(scenario, "july-week-02", True),
                block(scenario, "august-week-01", True),
            ]
            return undamaged_blocks + damaged_blocks


def load_template(template_path: Path) -> jinja2.Template:
    loader = jinja2.FileSystemLoader(searchpath=template_path.parent)
    template_environment = jinja2.Environment(loader=loader)
    return template_environment.get_template(template_path.name)


def render_template_and_save(template: jinja2.Template, output_path: Path, **kwargs):
    output_path.write_text(template.render(**kwargs))


def get_output_path_config(scenario: str, strategy: str, aggregation: str, mode: str):
    match scenario:
        case "N":
            return Path(
                f"configs/pbim/{scenario}/{mode}/merge-{aggregation}-{strategy}.json"
            )
        case _:
            return Path(
                f"configs/pbim/anomalous/{scenario}/merge-{aggregation}-{strategy}.json"
            )


def get_dataset_output_path(scenario: str, strategy: str, aggregation: str, mode: str):
    match scenario:
        case "N":
            return f"/data/PBIM/{scenario}/merged/reference/{mode}/{aggregation}-{strategy}/assembled.dat"
        case _:
            return f"/data/PBIM/{scenario}/merged/anomalous/reference/{aggregation}-{strategy}/assembled.dat"


def render_for_all_aggregations_and_strategies(
    scenario: str,
    job_template: jinja2.Template,
    merge_config_template: jinja2.Template,
    mode: str,
):
    for strategy in ["hourly", "uniform", "weighted-random"]:
        for aggregation in ["nosampling", "mean", "interpolate"]:
            output_path_job = Path(
                f"k8s/assemble_jobs/pbim/post-process-jobs/{scenario}/merge/{mode}/merge-{aggregation}-{strategy}.yml"
            )
            output_path_config = get_output_path_config(
                scenario, strategy, aggregation, mode
            )
            config_path_parameter = Path("/app") / output_path_config
            output_path_config.parent.mkdir(parents=True, exist_ok=True)
            output_path_job.parent.mkdir(parents=True, exist_ok=True)
            render_template_and_save(
                merge_config_template,
                output_path_config,
                STRATEGY=strategy,
                OUTPUT_FILE=get_dataset_output_path(
                    scenario, strategy, aggregation, mode
                ),
                AGGREGATION=aggregation,
                SCENARIO=scenario,
                FILES=json.dumps(
                    get_files(scenario, strategy, aggregation, mode), indent=4
                ),
            )
            render_template_and_save(
                job_template,
                output_path_job,
                STRATEGY=strategy,
                AGGREGATION=aggregation,
                SCENARIO=scenario.lower(),
                CONFIG_PATH=config_path_parameter,
            )


def main():
    job_template = load_template(Path("template/merge_pbim_job_template.yml"))
    merge_config_template = load_template(
        Path("template/merge_pbim_config_template.json")
    )
    # Training data
    render_for_all_aggregations_and_strategies(
        "N", job_template, merge_config_template, mode="train"
    )
    # Validation data
    render_for_all_aggregations_and_strategies(
        "N", job_template, merge_config_template, mode="val"
    )
    # Threshold data
    render_for_all_aggregations_and_strategies(
        "N", job_template, merge_config_template, mode="threshold"
    )
    # Test data
    for scenario in ["S1", "S2", "S3"]:
        render_for_all_aggregations_and_strategies(
            scenario, job_template, merge_config_template, mode="test"
        )


if __name__ == "__main__":
    main()
