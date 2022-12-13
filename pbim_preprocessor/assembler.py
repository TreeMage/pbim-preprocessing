import datetime
import struct
from pathlib import Path
from typing import List, Optional, BinaryIO, Generator, Dict, Any

from pbim_preprocessor.model import Measurement
from pbim_preprocessor.parser import POST_PROCESSABLE_CHANNELS
from pbim_preprocessor.processor import MEASUREMENT_SIZE_IN_BYTES
from pbim_preprocessor.sampling import SamplingStrategy


class Assembler:
    def __init__(self, sampling_strategy: SamplingStrategy, resolution: int):
        """
        :param sampling_strategy: The strategy to use for sampling the data
        :param resolution: The resolution to use for the sampling window in seconds.
        For instance, choosing a resolution of 60 will create a time-series that contains
        a value for every minute.
        """
        self._sampling_strategy = sampling_strategy
        self._resolution = resolution

    def assemble(
        self,
        path: Path,
        channels: Optional[List[str]] = None,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
    ) -> Generator[Dict[str, float], Any, None]:
        if not start_time:
            start_time = datetime.datetime.min
        if not end_time:
            end_time = datetime.datetime.max
        if not channels:
            channels = POST_PROCESSABLE_CHANNELS

        handles = {
            channel: self._prepare_file_handle(path, start_time, channel)
            for channel in channels
        }

        steps = int((end_time - start_time).total_seconds() / self._resolution)
        window_start = start_time
        window_end = window_start + datetime.timedelta(seconds=self._resolution / 2)
        for _ in range(1, steps):
            target = window_start + (window_end - window_start) / 2
            yield {
                channel: self._process_channel(f, window_start, window_end, target)
                for channel, f in handles.items()
            }
            window_start = window_end
            window_end = window_start + datetime.timedelta(seconds=self._resolution)

    def _prepare_file_handle(self, path: Path, time: datetime.datetime, channel: str):
        f = open(self._make_file_path(path, time, channel), "rb")
        # find the first timestamp in the file
        t0 = self._read_timestamp(f)
        t1 = self._read_timestamp(f)
        f.seek(-MEASUREMENT_SIZE_IN_BYTES, 2)
        t_final = self._read_timestamp(f)
        # check if we need to change file
        if time < t0:
            f.close()
            f = open(
                self._make_file_path(path, time - datetime.timedelta(days=1), channel),
                "rb",
            )
        elif time > t_final:
            f.close()
            f = open(
                self._make_file_path(path, time + datetime.timedelta(days=1), channel),
                "rb",
            )
        else:
            f.seek(0)
        # seek to the correct time (last measurement prior to the target one)
        print(f"Preparing handle for {channel}")
        return self._jump_to(f, time, t0, t1 - t0)

    def _process_channel(
        self, f: BinaryIO, start: datetime.datetime, end: datetime.datetime, time: datetime.datetime
    ):
        # Ensure we are at the start
        current_time = self._read_timestamp(f)
        f.seek(-MEASUREMENT_SIZE_IN_BYTES, 1)
        if current_time > start:
            self._find_linear(f, start, forward=False)

        values = [m for m in self._read_until(f, end)]
        return self._sampling_strategy.sample(values, time)

    def _jump_to(self, f: BinaryIO, time: datetime.datetime, t0: datetime.datetime, assumed_step: datetime.timedelta) -> BinaryIO:
        steps = int((time - t0) / assumed_step)
        f.seek(steps * MEASUREMENT_SIZE_IN_BYTES)
        current_time = self._read_timestamp(f)
        if current_time == time:
            return f
        else:
            f.seek(-MEASUREMENT_SIZE_IN_BYTES, 1)
            return self._find_linear(f, time, forward=current_time < time)

    def _find_linear(self, f: BinaryIO, time: datetime.datetime, forward=True) -> BinaryIO:
        current_time = self._read_timestamp(f)
        while current_time < time if forward else current_time > time:
            if not forward:
                f.seek(-2*MEASUREMENT_SIZE_IN_BYTES,1)
            current_time = self._read_timestamp(f)
        f.seek(-MEASUREMENT_SIZE_IN_BYTES, 1)
        return f

    def _read_until(self, f: BinaryIO, time: datetime.datetime) -> Generator[Measurement, Any, None]:
        current = self._read_measurement(f)
        yield current
        while current.time / 1000 < time.timestamp():
            current = self._read_measurement(f)
            yield current

    @staticmethod
    def _make_file_path(path: Path, time: datetime.datetime, channel: str):
        year, month, day = time.year, time.month, time.day
        return path / f"{year}" / f"{month}" / f"{day}" / f"{channel}.dat"

    @staticmethod
    def _read_timestamp(f: BinaryIO) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(
            struct.unpack("<q", f.read(MEASUREMENT_SIZE_IN_BYTES)[:8])[0] / 1000
        )

    @staticmethod
    def _read_measurement(f: BinaryIO) -> Measurement:
        time, value = struct.unpack("<qf", f.read(MEASUREMENT_SIZE_IN_BYTES))
        return Measurement(measurement=value, time=time)

    @staticmethod
    def _approximate_step(f: BinaryIO, samples: int = 100) -> datetime.timedelta:
        current = f.tell()
        data = f.read(samples * MEASUREMENT_SIZE_IN_BYTES)
        t0 = datetime.datetime.fromtimestamp(struct.unpack("<q", data[:8])[0] / 1000)
        t1 = datetime.datetime.fromtimestamp(struct.unpack("<q", data[samples * MEASUREMENT_SIZE_IN_BYTES - 8:])[0] / 1000)
        return None
