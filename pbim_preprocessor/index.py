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


def _write_index_file(path: Path, entries: List[CutIndexEntry]):
    with open(path.parent / f"{path.stem}.index.json", "w") as f:
        json.dump(
            [entry.to_dict() for entry in entries],
            f,
            indent=4,
        )


def _write_index(
    measurements: List[int],
    anomalous: List[bool] | bool,
    output_path: Path,
    original_lengths: List[int],
    existing_indices: List[List[CutIndexEntry]] | None = None,
    offsets: List[int] | None = None,
):
    if offsets is None:
        offsets = [0] * len(measurements)
    if existing_indices is None:
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
        i = 0
        current_offset = 0
        while i < len(measurements):
            num_measurements = measurements[i]
            index = existing_indices[i]
            offset = offsets[i]
            length = original_lengths[i]
            start_measurement_index = int(offset * length)
            end_measurement_index = start_measurement_index + num_measurements
            for entry in index:
                # Sample is before the cut
                if (
                    entry.start_measurement_index < start_measurement_index
                    and entry.end_measurement_index < end_measurement_index
                ):
                    continue
                # Sample is after the cut
                if entry.start_measurement_index > end_measurement_index:
                    continue
                # Sample intersects start of the cut
                if (
                    entry.start_measurement_index
                    < start_measurement_index
                    < entry.end_measurement_index
                ):
                    entry.start_measurement_index = start_measurement_index
                # Sample intersects end of the cut
                if (
                    entry.start_measurement_index
                    < end_measurement_index
                    < entry.end_measurement_index
                ):
                    entry.end_measurement_index = end_measurement_index

                entry.start_measurement_index += current_offset
                entry.end_measurement_index += current_offset
                entry.anomalous = (
                    anomalous[i] if isinstance(anomalous, list) else anomalous
                )
                entries.append(entry)
            i += 1
            current_offset += num_measurements

    _write_index_file(output_path, entries)
