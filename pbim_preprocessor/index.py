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


def _build_index_windowed(
    windows: List[int],
    anomalous: List[bool] | bool,
    existing_indices: List[CutIndex],
    offsets: List[int],
):
    entries = []
    i = 0
    current_offset = 0
    while i < len(windows):
        current_index = existing_indices[i]
        index_offset = offsets[i]
        current_start_index = int(len(current_index.entries) * index_offset)
        current_end_index = current_start_index + windows[i]
        for index_entry in current_index.entries[current_start_index:current_end_index]:
            length = (
                index_entry.end_measurement_index - index_entry.start_measurement_index
            )
            start = (
                index_entry.start_measurement_index
                + current_offset
                - current_index.entries[current_start_index].start_measurement_index
            )
            entries.append(
                CutIndexEntry(
                    start,
                    start + length,
                    anomalous[i] if isinstance(anomalous, list) else anomalous,
                )
            )
        block_length = (
            current_index.entries[current_end_index - 1].end_measurement_index
            - current_index.entries[current_start_index].start_measurement_index
        )
        current_offset += block_length
        i += 1
    return CutIndex(True, entries)


def _build_index_contiguous(
    measurements: List[int],
    anomalous: List[bool] | bool,
    original_lengths: List[int],
    existing_indices: List[CutIndex],
    offsets: List[int],
):
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
            entry.anomalous = anomalous[i] if isinstance(anomalous, list) else anomalous
            entries.append(entry)
        i += 1
        current_offset += num_measurements
    return CutIndex(False, entries)


def _write_index(
    output_path: Path,
    is_window_index: bool,
    measurements_or_windows: List[int],
    anomalous: List[bool] | bool,
    original_lengths: List[int] | None,
    existing_indices: List[CutIndex] | None,
    offsets: List[int] | None = None,
):
    if offsets is None:
        offsets = [0] * len(measurements_or_windows)
    if is_window_index:
        index = _build_index_windowed(
            measurements_or_windows, anomalous, existing_indices, offsets
        )
    else:
        if original_lengths is None:
            raise ValueError("original_lengths must be provided for contiguous indices")
        if existing_indices is None:
            if existing_indices is None:
                existing_indices = [
                    CutIndex(
                        False,
                        [
                            CutIndexEntry(
                                0,
                                end,
                                anomalous[i]
                                if isinstance(anomalous, list)
                                else anomalous,
                            )
                        ],
                    )
                    for i, end in enumerate(measurements_or_windows)
                ]
        index = _build_index_contiguous(
            measurements_or_windows,
            anomalous,
            original_lengths,
            existing_indices,
            offsets,
        )
    _write_index_file(output_path, index)
