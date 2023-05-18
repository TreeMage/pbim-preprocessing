import datetime
from pathlib import Path

import jinja2

# First thursday of every month
START_TIMES = {
    "april-week-": datetime.datetime(2018, 4, 5),
    "january-week-": datetime.datetime(2018, 1, 4),
    "june-week-": datetime.datetime(2018, 6, 7),
    "may-week-": datetime.datetime(2018, 5, 3),
    "february-week-": datetime.datetime(2018, 2, 1),
    "march-week-": datetime.datetime(2018, 3, 1),
    "july-week-": datetime.datetime(2018, 7, 5),
    "august-week-": datetime.datetime(2018, 8, 2),
}

DURATION = datetime.timedelta(days=3, hours=23, minutes=59, seconds=59)

NORMAL_FILES = [
    "april-week-",
    "january-week-",
    "june-week-",
    "may-week-",
    "february-week-",
    "march-week-",
]

DAMAGED_FILES = [
    "july-week-",
    "august-week-",
]


def load_template(template_path: Path) -> jinja2.Template:
    loader = jinja2.FileSystemLoader(searchpath=template_path.parent)
    template_environment = jinja2.Environment(loader=loader)
    return template_environment.get_template(template_path.name)


def render_template_and_save(template: jinja2.Template, output_path: Path, **kwargs):
    output_path.write_text(template.render(**kwargs))


def render_for_all_aggregations(
    file_name_template: str, week: int, scenario: str, template: jinja2.Template
):
    file_name = f"{file_name_template}{week:02d}"
    start_time = START_TIMES[file_name_template] + DURATION * (week - 1)
    end_time = start_time + DURATION
    for aggregation in ["mean", "interpolate", "nosampling"]:
        output_path = Path(
            f"k8s/assemble_jobs/pbim/artificial/{scenario}/{aggregation}/{file_name}.yml"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        render_template_and_save(
            template,
            output_path,
            scenario=scenario,
            filename=file_name,
            strategy=aggregation if aggregation != "nosampling" else "mean",
            start_time=start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            end_time=end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            resolution=0 if aggregation == "nosampling" else 0.04,
        )


def main():
    template = load_template(Path("template/assemble_job_template_pbim.yml"))
    scenario = "N"
    for file_name_template in NORMAL_FILES:
        for week in [1, 2]:
            render_for_all_aggregations(file_name_template, week, scenario, template)

    for scenario in ["S1", "S2", "S3"]:
        # July
        file_name_template = "july-week-"
        week = 2
        render_for_all_aggregations(file_name_template, week, scenario, template)
        # August
        file_name_template = "august-week-"
        week = 1
        render_for_all_aggregations(file_name_template, week, scenario, template)


if __name__ == "__main__":
    main()
