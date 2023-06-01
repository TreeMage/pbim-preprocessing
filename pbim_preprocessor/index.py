from dataclasses import dataclass
from pathlib import Path
from typing import List

import struct


CUT_INDEX_ENTRY_SIZE = 17


@dataclass
class CutIndexEntry:
    start_measurement_index: int
    end_measurement_index: int
    anomalous: bool

    def to_bytes(self) -> bytes:
        return struct.pack(
            "<qq?",
            self.start_measurement_index,
            self.end_measurement_index,
            self.anomalous,
        )

    @staticmethod
    def from_bytes(data: bytes) -> "CutIndexEntry":
        start, end, anomalous = struct.unpack("<qq?", data)
        return CutIndexEntry(start, end, anomalous)


@dataclass
class CutIndex:
    is_window_index: bool
    entries: List[CutIndexEntry]
    version: int = 1

    def to_bytes(self) -> bytes:
        return struct.pack("<?i", self.is_window_index, self.version) + b"".join(
            [entry.to_bytes() for entry in self.entries]
        )

    @staticmethod
    def from_bytes(data: bytes) -> "CutIndex":
        entries = []
        (is_window_index, version) = struct.unpack("<?i", data[:5])
        for i in range(5, len(data), CUT_INDEX_ENTRY_SIZE):
            entries.append(CutIndexEntry.from_bytes(data[i : i + CUT_INDEX_ENTRY_SIZE]))
        return CutIndex(is_window_index, entries, version)


def _write_index_file(path: Path, index: CutIndex):
    serialized = index.to_bytes()
    with open(path.parent / f"{path.stem}.index", "wb") as f:
        f.write(serialized)


def _write_index(
    measurements: List[int],
    anomalous: List[bool] | bool,
    output_path: Path,
    original_lengths: List[int],
    existing_indices: List[CutIndex] | None = None,
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
            for entry in index.entries:
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
                # Sample is fully contained in the cut
                # Adjust index to offset in the input file
                entry.start_measurement_index -= start_measurement_index
                entry.end_measurement_index -= start_measurement_index
                # Adjust index to offset due to merging
                entry.start_measurement_index += current_offset
                entry.end_measurement_index += current_offset
                entry.anomalous = (
                    anomalous[i] if isinstance(anomalous, list) else anomalous
                )
                entries.append(entry)
            i += 1
            current_offset += num_measurements

    final_index = CutIndex(entries)
    _write_index_file(output_path, final_index)
