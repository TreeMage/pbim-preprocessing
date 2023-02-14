import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class CutIndexEntry:
    start_measurement_index: int
    end_measurement_index: int
    anomalous: bool


def _write_index(
    measurements: List[int], anomalous: List[bool] | bool, output_path: Path
):
    indices = [0] + np.cumsum(measurements).tolist()
    entries = [
        CutIndexEntry(
            start,
            end,
            anomalous[i] if isinstance(anomalous, list) else anomalous,
        )
        for i, (start, end) in enumerate(zip(indices[:-1], indices[1:]))
    ]
    with open(output_path, "w") as f:
        json.dump(
            [entry.to_dict() for entry in entries],
            f,
            indent=4,
        )
