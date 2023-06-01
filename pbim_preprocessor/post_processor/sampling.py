import abc
import calendar
import datetime
import math
import random

from typing import List, Tuple

import numpy as np


def _available_windows(start: int, end: int, window_size: int):
    return max(end - start - window_size + 1, 0)


def _available_windows_total(
    start_and_end_indices: List[Tuple[int, int]], window_size: int
):
    return sum(
        _available_windows(start, end, window_size)
        for start, end in start_and_end_indices
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


def _sample_next_interval(
    windows_per_sample: int,
    window_size: int,
    start_and_end_indices: List[Tuple[int, int]],
    start_index: int,
) -> Tuple[List[Tuple[int, int]], int] | None:
    if (group_index := _find_group_index(start_index, start_and_end_indices)) is None:
        return None

    sample_indices = []
    current_length_in_windows = 0
    current_start, current_end = start_and_end_indices[group_index]
    if start_index < current_start:
        start_index = current_start
    current_index = start_index
    while current_length_in_windows < windows_per_sample:
        windows_in_current_cut = current_end - current_start - window_size + 1
        if current_length_in_windows + windows_in_current_cut <= windows_per_sample:
            sample_indices.extend(
                [
                    (current_start + i, current_start + i + window_size)
                    for i in range(windows_in_current_cut)
                ]
            )
            current_index = current_end
            current_length_in_windows += windows_in_current_cut
        else:
            left_over_windows = windows_per_sample - current_length_in_windows
            sample_indices.extend(
                [
                    (current_start + i, current_start + i + window_size)
                    for i in range(left_over_windows)
                ]
            )
            current_index += left_over_windows
            current_length_in_windows += left_over_windows
        if current_index >= current_end:
            group_index += 1
            if group_index >= len(start_and_end_indices):
                break
            current_start, current_end = start_and_end_indices[group_index]
            current_index = current_start

    return sample_indices, current_index


class DatasetSamplingStrategy(abc.ABC):
    @abc.abstractmethod
    def compute_sample_indices(
        self, time: np.ndarray, start_and_end_indices: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        pass


class AlternatingRound:
    def __init__(self):
        self.up = True

    def reset(self):
        self.up = True

    def __call__(self, value: float) -> int:
        if self.up:
            rounded = math.ceil(value)
        else:
            rounded = math.floor(value)
        self.up = not self.up
        return rounded


class UniformSamplingStrategy(DatasetSamplingStrategy):
    def __init__(self, num_windows: int, window_size: int):
        self._num_windows = num_windows
        self._window_size = window_size

    def compute_sample_indices(
        self, time: np.ndarray, start_and_end_indices: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        available_windows = _available_windows_total(
            start_and_end_indices, self._window_size
        )
        if available_windows < self._num_windows:
            raise ValueError(
                f"Cannot sample {self._num_windows} from {available_windows} windows."
            )
        alterating_round = AlternatingRound()
        windows_per_group = [
            max(
                alterating_round(
                    (end - start - self._window_size)
                    * self._num_windows
                    / available_windows
                ),
                1,
            )
            for start, end in start_and_end_indices
        ]
        window_start_and_end_indices = []
        for (start, end), desired_windows in zip(
            start_and_end_indices, windows_per_group
        ):
            max_window_start_index = end - self._window_size
            step = max(1, (max_window_start_index - start) // desired_windows)
            for i in range(desired_windows):
                window_start_and_end_indices.append(
                    (start + i * step, start + i * step + self._window_size)
                )
        return window_start_and_end_indices


class WeightedRandomSamplingStrategy(DatasetSamplingStrategy):
    def __init__(self, num_windows: int, window_size: int):
        self._num_windows = num_windows
        self._window_size = window_size

    @staticmethod
    def _compute_weight(timestamp: int) -> float:
        time = datetime.datetime.utcfromtimestamp(timestamp / 1000)
        if 6 <= time.hour <= 9:
            return 1
        if 16 <= time.hour <= 19:
            return 1
        if 20 <= time.hour <= 5:
            return 0.25
        return 0.5

    def compute_sample_indices(
        self, time: np.ndarray, start_and_end_indices: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        available_windows = _available_windows_total(
            start_and_end_indices, self._window_size
        )
        if available_windows < self._num_windows:
            raise ValueError(
                f"Cannot sample {self._num_windows} from {available_windows} samples."
            )
        window_indices = []
        for start, end in start_and_end_indices:
            window_indices.extend(list(range(start, end - self._window_size)))
        weights = np.array(
            [self._compute_weight(time[index]) for index in window_indices]
        )
        weights = weights / np.sum(weights)
        sampled_indices = np.random.choice(
            window_indices,
            self._num_windows,
            replace=False,
            p=weights,
        ).tolist()
        return [(index, index + self._window_size) for index in sampled_indices]


class IntervalSamplingStrategy(DatasetSamplingStrategy):
    def __init__(
        self,
        interval_length_in_seconds: int,
        samples_per_interval: int,
        windows_per_sample: int,
        window_size: int,
    ):
        self._interval_length_in_seconds = interval_length_in_seconds
        self._samples_per_interval = samples_per_interval
        self._windows_per_sample = windows_per_sample
        self._window_size = window_size

    @staticmethod
    def _make_datetime(timestamp: float) -> datetime.datetime:
        return datetime.datetime.utcfromtimestamp(timestamp / 1000)

    @staticmethod
    def _make_timestamp(dt: datetime.datetime) -> float:
        return calendar.timegm(dt.timetuple()) * 1000

    def compute_sample_indices(
        self, time: np.ndarray, start_and_end_indices: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        final_indices = []
        start_date = self._make_datetime(time[start_and_end_indices[0][0]])
        end_date = self._make_datetime(time[start_and_end_indices[-1][1]] - 1)
        current_date = start_date
        while current_date < end_date - datetime.timedelta(
            seconds=self._interval_length_in_seconds
        ):
            for i in range(self._samples_per_interval):
                interval_start = current_date + datetime.timedelta(
                    seconds=i
                    * self._interval_length_in_seconds
                    / self._samples_per_interval
                )
                if interval_start > end_date:
                    break
                start_index = np.searchsorted(
                    time, self._make_timestamp(interval_start)
                ).item()
                interval = _sample_next_interval(
                    self._windows_per_sample,
                    self._window_size,
                    start_and_end_indices,
                    start_index,
                )
                if not interval:
                    break
                sample_indices, end_index = interval
                final_indices.extend(sample_indices)
            current_date += datetime.timedelta(seconds=self._interval_length_in_seconds)
        return final_indices


class HourlySamplingStrategy(IntervalSamplingStrategy):
    def __init__(
        self, samples_per_hour: int, windows_per_sample: int, window_size: int
    ):
        super().__init__(3600, samples_per_hour, windows_per_sample, window_size)


if __name__ == "__main__":
    strat = UniformSamplingStrategy(666666, 256)
    indices = []
    current_start = 0
    for i in range(259000):
        gap = random.randint(1, 300)
        length = random.randint(1000, 2000)
        indices.append((current_start + gap, current_start + gap + length))
        current_start += gap + length
    print(len(strat.compute_sample_indices(np.arange(1), indices)))
