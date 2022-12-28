import abc
import datetime
from statistics import mean
from typing import List

from pbim_preprocessor.model import Measurement


class SamplingStrategy(abc.ABC):
    @abc.abstractmethod
    def sample(self, data: List[Measurement], target: datetime.datetime) -> float:
        pass


class MeanSamplingStrategy(SamplingStrategy):
    def sample(self, data: List[Measurement], target: datetime.datetime) -> float:
        return mean([m.measurement for m in data])


class LinearInterpolationSamplingStrategy(SamplingStrategy):
    def sample(self, data: List[Measurement], target: datetime.datetime) -> float:
        timestamp = int(target.timestamp()) * 1000
        left = self._find_left(data, timestamp)
        right = self._find_right(data, timestamp)
        if left.time == right.time:
            return left.measurement
        t = (timestamp - left.time) / (right.time - left.time)
        return left.measurement + t * (right.measurement - left.measurement)

    @staticmethod
    def _find_left(data: List[Measurement], target_timestamp: int) -> Measurement:
        current_index = 0
        while data[current_index].time < target_timestamp and current_index < len(data):
            current_index += 1
        return data[max(0, current_index - 1)]

    @staticmethod
    def _find_right(data: List[Measurement], target_timestamp: int) -> Measurement:
        current_index = len(data) - 1
        while data[current_index].time > target_timestamp and current_index > 0:
            current_index -= 1
        return data[min(len(data) - 1, current_index + 1)]
