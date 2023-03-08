import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class CutIndexEntry:
    start_measurement_index: int
    end_measurement_index: int
    anomalous: bool


def _write_index(
    measurements: List[int],
    anomalous: List[bool] | bool,
    output_path: Path,
    exsiting_indices: List[List[CutIndexEntry]] | None = None,
):
    if exsiting_indices is None:
        indices = [0] + measurements
        entries = [
            CutIndexEntry(
                start,
                end,
                anomalous[i] if isinstance(anomalous, list) else anomalous,
            )
            for i, (start, end) in enumerate(zip(indices[:-1], indices[1:]))
        ]
    else:
        entries = []
        for offset, indices in zip([0] + measurements, exsiting_indices):
            for entry in indices:
                entry.start_measurement_index += offset
                entry.end_measurement_index += offset
                entries.append(entry)

    with open(output_path, "w") as f:
        json.dump(
            [entry.to_dict() for entry in entries],
            f,
            indent=4,
        )
