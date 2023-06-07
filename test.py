import itertools

from pathlib import Path

import jinja2


def load_template(template_path: Path) -> jinja2.Template:
    loader = jinja2.FileSystemLoader(searchpath=template_path.parent)
    template_environment = jinja2.Environment(loader=loader)
    return template_environment.get_template(template_path.name)


def render_template_and_save(template: jinja2.Template, output_path: Path, **kwargs):
    output_path.write_text(template.render(**kwargs))


template = load_template(Path("template/postprocess_lux_job_template.yml"))

combinations = itertools.product([64, 128, 256, 512], [6, 8, 10], [100, 200, 300])
commands = []
for (grace_period, top_k, max_freq) in combinations:
    file_name = f"grace-period-{grace_period}-top-k-{top_k}-max-freq-{max_freq}"
    output_file = f"/data/LUX/N/filter/{file_name}/assembled.dat"
    job_output_path = Path(f"k8s/filter_lux/filter-{file_name}.yml")
    job_output_path.parent.mkdir(parents=True, exist_ok=True)
    render_template_and_save(
        template,
        job_output_path,
        SEED=42,
        INPUT_FILE="/data/LUX/N/assembled/nosampling/assembled.dat",
        OUTPUT_FILE=output_file,
        STRATEGY="noop",
        SCENARIO=f"{grace_period}-{top_k}-{max_freq}",
        STRATEGY_ARGS=(
            f"--window-size 512 "
            f"--sampling-rate 2500 "
            f"--lower-bound-first-frequency 2 "
            f"--upper-bound-first-frequency 30 "
            f"--lower-bound 0 "
            f"--upper-bound {max_freq} "
            f"--top-k {top_k} "
            f"--grace-period {grace_period} "
        ),
    )
