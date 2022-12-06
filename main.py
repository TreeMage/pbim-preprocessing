import json
import sys
from pathlib import Path

import matplotlib.pyplot

from parser.parser import PBimParser, POST_PROCESSABLE_CHANNELS


def main():
    path = Path(sys.argv[1])
    name = sys.argv[2]
    parser = PBimParser()
    data = parser.parse(path, name)
    for channel, measurements in data.items():
        if channel.name in POST_PROCESSABLE_CHANNELS:
            with open(f"../data/parsed/{channel.name}.json", "w") as f:
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


def plot():
    import matplotlib.pyplot as plt
    import json

    for channel in POST_PROCESSABLE_CHANNELS:
        with open(f"../data/parsed/{channel}.json") as f:
            data = json.load(f)
            fig, ax = plt.subplots()
            ax.plot(data["time"], data["data"])
            ax.set_title(channel)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(data["metadata"]["unit"])
            fig.savefig(f"../data/plots/{channel}.png")
            matplotlib.pyplot.close(fig)


if __name__ == "__main__":
    main()
    plot()
