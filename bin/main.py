import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import click
import matplotlib.pyplot

# I hate this, but I have no idea to how to make it work otherwise.
sys.path.append(str(Path(__file__).parent.parent))
from pbim_preprocessor.parser import PBimParser, POST_PROCESSABLE_CHANNELS


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument(
    "output-path", type=click.Path(writable=True, file_okay=False, path_type=Path)
)
@click.argument("name")
def parse(path: Path, output_path: Path, name: str):
    """
    Parse data from a Job specified by NAME located in the directory specified by PATH. Results are stored in the directory
    specified by OUTPUT_PATH.
    """
    output_path.mkdir(exist_ok=True, parents=True)
    parser = PBimParser()
    data = parser.parse(path, name)
    for channel, measurements in data.items():
        if channel.name in POST_PROCESSABLE_CHANNELS:
            with open(output_path / f"{channel.name}.json", "w") as f:
                json.dump(
                    {
                        "metadata": channel.to_dict(),
                        "data": [
                            measurement.measurement for measurement in measurements
                        ],
                        "time": [measurement.time for measurement in measurements],
                    },
                    f,
                )


@click.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.argument("output-path", type=click.Path(path_type=Path))
@click.option(
    "--file",
    default=None,
    help="Name of the file to plot. If not specified, all files in the input directory will be plotted.",
)
def plot(path: Path, output_path: Path, file: Optional[str]):
    """
    Plot sensor data by parsing a json files located in the directory specified by PATH. Results are stored in the
    directory specified by OUTPUT_PATH.
    If no FILE is provided, all files in the input directory will be plotted.
    """
    import matplotlib.pyplot as plt
    import json

    def plot_single(data: Dict[str, Any], name: str, output_path: Path):
        fig, ax = plt.subplots()
        ax.plot(data["time"], data["data"])
        ax.set_title(name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(data["metadata"]["unit"])
        fig.savefig(output_path / f"{name}.png")
        matplotlib.pyplot.close(fig)

    if file:
        file_path = path / file
        plot_single(json.load(open(file_path)), file_path.name, output_path)
    else:
        for file in path.iterdir():
            if file.name.endswith(".json"):
                plot_single(json.load(open(file)), file.name, output_path)


@click.group()
def cli():
    pass


cli.add_command(parse)
cli.add_command(plot)

if __name__ == "__main__":
    cli()
