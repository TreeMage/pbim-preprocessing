import math
from typing import List, Tuple

from pbim_preprocessor.index import CutIndex


def available_windows(start: int, end: int, window_size: int):
    return max(end - start - window_size + 1, 0)


def available_windows_total(
    indices: CutIndex | List[Tuple[int, int]], window_size: int
) -> int:
    if isinstance(indices, CutIndex):
        return _available_windows_total_index(indices, window_size)
    elif isinstance(indices, list):
        return _available_windows_total_start_end_indices(indices, window_size)
    else:
        raise TypeError(f"Unknown type {type(indices)}")


def _available_windows_total_start_end_indices(
    indices: List[Tuple[int, int]], window_size: int
) -> int:
    return sum(available_windows(start, end, window_size) for start, end in indices)


def _available_windows_total_index(indices: CutIndex, window_size: int) -> int:
    return sum(
        available_windows(
            entry.start_measurement_index, entry.end_measurement_index, window_size
        )
        for entry in indices.entries
    )


def _find_group_index(index: int, start_and_end_indices: List[Tuple[int, int]]):
    for i, (start, end) in enumerate(start_and_end_indices):
        if start <= index < end:
            return i

    end_start_pairs = [(0, start_and_end_indices[0][0])]
    for i in range(len(start_and_end_indices) - 1):
        first_start, first_end = start_and_end_indices[i]
        second_start, second_end = start_and_end_indices[i + 1]
        end_start_pairs.append((first_end, second_start))

    for i, (end, start) in enumerate(end_start_pairs):
        if end <= index < start:
            return i

    return None


def _merge_windows(sample_indices: List[int], window_size: int):
    last_index = -math.inf
    final_indices = []
    for index in sample_indices:
        if index < last_index + window_size:
            adjusted_index = last_index + window_size
            remaining_length = index - last_index
        else:
            adjusted_index = index
            remaining_length = window_size
        last_index = index
        final_indices.extend([adjusted_index + i for i in range(remaining_length)])
    return final_indices


class RoundingWithFractionalTracker:
    def __init__(self):
        self.fractional_part = 0.0

    def reset(self):
        self.fractional_part = 0.0

    def __call__(self, value: float) -> int:
        new_fractional_part = self.fractional_part + value - math.floor(value)
        floored = math.floor(value)
        if new_fractional_part >= 1:
            new_fractional_part -= 1
            floored += 1
        self.fractional_part = new_fractional_part
        return floored
