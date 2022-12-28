import datetime
import struct
from pathlib import Path
from typing import List, Optional, BinaryIO, Generator, Dict, Any, Tuple

from pbim_preprocessor.model import Measurement
from pbim_preprocessor.parser import POST_PROCESSABLE_CHANNELS
from pbim_preprocessor.processor import MEASUREMENT_SIZE_IN_BYTES
from pbim_preprocessor.sampling import SamplingStrategy
from pbim_preprocessor.utils import GeneratorWithReturnValue, LOGGER


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
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        channels: Optional[List[str]] = None,
    ) -> Generator[Dict[str, float], Any, None]:
        if not channels:
            channels = POST_PROCESSABLE_CHANNELS
        LOGGER.info(
            f"Assembling data from {start_time} to {end_time} with a resolution of {self._resolution}s."
        )
        LOGGER.info(f"Using {len(channels)} channels.")

        handles = {
            channel: self._prepare_file_handle(path, start_time, channel)
            for channel in channels
        }

        approximate_steps = {
            channel: self._approximate_step(handles[channel]) for channel in channels
        }

        steps = int((end_time - start_time).total_seconds() / self._resolution)
        window_start = start_time
        window_end = window_start + datetime.timedelta(seconds=self._resolution / 2)
        for i in range(steps):
            target = window_start + (window_end - window_start) / 2
            LOGGER.info(f"Processing step {i+1} of {steps}. Current time: {target}.")
            data = {"time": target.timestamp()}
            for channel, handle in handles.items():
                approximate_step = approximate_steps[channel]
                done, current_time, values = self._process_channel(
                    handle, window_start, window_end, approximate_step=approximate_step
                )
                if not done:
                    LOGGER.info(
                        f"Current file exhausted at time {current_time}. Preparing new file handle.",
                        identifier=channel,
                    )
                    new_handle = self._prepare_file_handle(
                        path,
                        current_time,
                        channel,
                        approximate_step=approximate_step,
                        previous_file_exhausted=True,
                    )
                    handles[channel] = new_handle
                    _, _, additional_values = self._process_channel(
                        new_handle,
                        current_time,
                        window_end,
                        approximate_step=approximate_step,
                    )
                    values += additional_values
                data[channel] = self._sampling_strategy.sample(values, target)
            yield data
            window_start = window_end
            window_end = window_start + datetime.timedelta(seconds=self._resolution)

    def _prepare_file_handle(
        self,
        path: Path,
        time: datetime.datetime,
        channel: str,
        approximate_step: Optional[datetime.timedelta] = None,
        previous_file_exhausted: bool = False,
    ) -> BinaryIO:
        LOGGER.info(f"Preparing file handle at time {time}.", identifier=channel)
        f = open(self._make_file_path(path, time, channel), "rb")
        # find the first timestamp in the file
        t0 = self._read_timestamp(f)
        f.seek(-MEASUREMENT_SIZE_IN_BYTES, 2)
        t_final = self._read_timestamp(f)
        LOGGER.info(f"Initial file spans {t0} - {t_final}.", identifier=channel)
        # check if we need to change file
        if time < t0:
            if previous_file_exhausted:
                LOGGER.warn(
                    "Target time stamp is to early for this file but the previous one is exhausted. There is data missing.",
                    identifier=channel,
                )
                LOGGER.warn(
                    "Using current time step as target instead.", identifier=channel
                )
                f.seek(0)
                return f

            LOGGER.info(f"Target too early. Switching files.", identifier=channel)
            path = self._make_file_path(
                path, time - datetime.timedelta(days=1), channel
            )
            LOGGER.info(f"New file handle: {path}", identifier=channel)
            f.close()
            f = open(path, "rb")
            t0 = self._read_timestamp(f)
        elif time > t_final:
            LOGGER.info(f"Target too late. Switching files.", identifier=channel)
            path = self._make_file_path(
                path, time + datetime.timedelta(days=1), channel
            )
            LOGGER.info(f"New file handle: {path}", identifier=channel)
            f.close()
            f = open(path, "rb")
            t0 = self._read_timestamp(f)
        # seek to the correct time (last measurement prior to the target one)
        LOGGER.info("Seeking to target offset.", identifier=channel)
        f.seek(0)
        if not approximate_step:
            approximate_step = self._approximate_step(f)
        LOGGER.info(f"Approximate step: {approximate_step}.", identifier=channel)
        return self._jump_to(f, time, t0, approximate_step or self._approximate_step(f))

    def _process_channel(
        self,
        f: BinaryIO,
        start: datetime.datetime,
        end: datetime.datetime,
        approximate_step: Optional[datetime.timedelta] = None,
    ):
        # Ensure we are at the start
        if not f.read(1):
            # There is no data left
            return False, start, []
        else:
            f.seek(-1, 1)

        current_time = self._read_timestamp(f)
        f.seek(-MEASUREMENT_SIZE_IN_BYTES, 1)
        # Only seek back if we are not at the beginning of the file. Necessary because of small gaps in the recording.
        if current_time > start:
            if f.tell() == 0:
                LOGGER.warn(
                    "File exhausted at start of file. This is probably caused by missing measurement data."
                )
            else:
                self._find_linear(f, start, forward=False)

        generator = GeneratorWithReturnValue(
            self._read_until_buffered(
                f, end, approximate_step or self._approximate_step(f)
            )
        )
        values = [m for m in generator]
        done = generator.value
        end_time = datetime.datetime.fromtimestamp(values[-1].time / 1000)
        return (
            done,
            end_time,
            values,
        )

    def _jump_to(
        self,
        f: BinaryIO,
        time: datetime.datetime,
        t0: datetime.datetime,
        assumed_step: datetime.timedelta,
    ) -> BinaryIO:
        steps = int((time - t0) / assumed_step)
        f.seek(steps * MEASUREMENT_SIZE_IN_BYTES)
        if f.read(1) == b"":
            LOGGER.warn(
                "Jumped over the end of the file. Seeking back to target linearly."
            )
            f.seek(-MEASUREMENT_SIZE_IN_BYTES, 2)
        else:
            f.seek(-1, 1)
        current_time = self._read_timestamp(f)
        if current_time == time:
            return f
        else:
            f.seek(-MEASUREMENT_SIZE_IN_BYTES, 1)
            return self._find_linear(f, time, forward=current_time < time)

    def _find_linear(
        self, f: BinaryIO, time: datetime.datetime, forward=True
    ) -> BinaryIO:
        current_time = self._read_timestamp(f)
        while current_time < time if forward else current_time > time:
            if not forward:
                f.seek(-2 * MEASUREMENT_SIZE_IN_BYTES, 1)
            current_time = self._read_timestamp(f)
        f.seek(-MEASUREMENT_SIZE_IN_BYTES, 1)
        return f

    def _read_until(
        self, f: BinaryIO, time: datetime.datetime
    ) -> Generator[Measurement, Any, bool]:
        try:
            current = self._read_measurement(f)
            yield current
            while current.time / 1000 < time.timestamp():
                current = self._read_measurement(f)
                yield current
            return True
        except struct.error:
            return False

    def _read_until_buffered(
        self, f: BinaryIO, time: datetime.datetime, approx_step: datetime.timedelta
    ) -> Generator[Measurement, Any, bool]:
        try:
            current = self._read_measurement(f)
            yield current
            num_measurements = int(
                (time - datetime.datetime.fromtimestamp(current.time / 1000))
                / approx_step
            )
            buffer = f.read(num_measurements * MEASUREMENT_SIZE_IN_BYTES)
            offset = 0
            while current.time / 1000 < time.timestamp():
                current = self._read_measurement_from_buffer(
                    buffer[offset : offset + MEASUREMENT_SIZE_IN_BYTES]
                )
                offset += MEASUREMENT_SIZE_IN_BYTES
                yield current
                if offset >= len(buffer):
                    return (yield from self._read_until(f, time))
            return True
        except struct.error:
            return False

    @staticmethod
    def _make_file_path(path: Path, time: datetime.datetime, channel: str):
        return path / f"{time:%Y}" / f"{time:%m}" / f"{time:%d}" / f"{channel}.dat"

    @staticmethod
    def _read_timestamp(f: BinaryIO) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(
            struct.unpack("<q", f.read(MEASUREMENT_SIZE_IN_BYTES)[:8])[0] / 1000
        )

    @staticmethod
    def _read_measurement_from_buffer(buffer: bytes) -> Measurement:
        time, value = struct.unpack("<qf", buffer)
        return Measurement(measurement=value, time=time)

    @staticmethod
    def _read_measurement(f: BinaryIO) -> Measurement:
        time, value = struct.unpack("<qf", f.read(MEASUREMENT_SIZE_IN_BYTES))
        return Measurement(measurement=value, time=time)

    @staticmethod
    def _approximate_step(f: BinaryIO, samples: int = 100) -> datetime.timedelta:
        current = f.tell()
        data = f.read(samples * MEASUREMENT_SIZE_IN_BYTES)
        f.seek(current)
        t0 = datetime.datetime.fromtimestamp(struct.unpack("<q", data[:8])[0] / 1000)
        t1 = datetime.datetime.fromtimestamp(
            struct.unpack("<q", data[samples * MEASUREMENT_SIZE_IN_BYTES - 12 : -4])[0]
            / 1000
        )
        return (t1 - t0) / samples
