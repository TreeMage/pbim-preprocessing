import datetime
import math
import struct
from pathlib import Path
from statistics import mean
from typing import (
    List,
    Optional,
    BinaryIO,
    Generator,
    Dict,
    Any,
    Tuple,
    TextIO,
    Literal,
)

from pbim_preprocessor.model import Measurement, EOF, ParsedZ24File
from pbim_preprocessor.parser import (
    POST_PROCESSABLE_CHANNELS,
    Z24UndamagedParser,
    Z24DamagedParser,
)
from pbim_preprocessor.processor import MEASUREMENT_SIZE_IN_BYTES
from pbim_preprocessor.sampling import SamplingStrategy
from pbim_preprocessor.utils import GeneratorWithReturnValue, LOGGER


class PBimAssembler:
    def __init__(
        self, path: Path, sampling_strategy: SamplingStrategy, resolution: float
    ):
        """
        :param path: Path to the data directory.
        :param sampling_strategy: The strategy to use for sampling the data
        :param resolution: The resolution to use for the sampling window in seconds.
        For instance, choosing a resolution of 60 will create a time-series that contains
        a value for every minute.
        """
        self._path = path
        self._sampling_strategy = sampling_strategy
        self._resolution = resolution

    def assemble(
        self,
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
        handles = {}
        for channel in channels:
            handle, time, stop = self._prepare_file_handle(start_time, channel)
            if stop:
                LOGGER.error(
                    f"Could not prepare file handle for start time {start_time}."
                )
                return None
            if time > start_time:
                start_time = time
            handles[channel] = handle
        LOGGER.info(f"Earliest time with data is {start_time}.")
        approximate_steps = {
            channel: self._approximate_step(handles[channel]) for channel in channels
        }
        steps = int((end_time - start_time).total_seconds() / self._resolution)
        window_start = start_time
        window_end = window_start + datetime.timedelta(seconds=self._resolution / 2)
        should_stop = False
        last_values = None
        for i in range(steps):
            if window_end > end_time:
                LOGGER.warn(
                    "Stopping early because end time is reached. This is caused by missing data."
                )
                break
            if should_stop:
                LOGGER.warn(
                    "Stopping early because all files are exhausted. This is caused by missing data."
                )
                break
            target = window_start + (window_end - window_start) / 2
            LOGGER.info(
                f"Processing step {i+1} of {steps}. Current time: {target.strftime('%Y-%m-%d %H:%M:%S')}."
            )
            LOGGER.debug(f"Window: {window_start} - {window_end} with target {target}.")
            data = {"time": target.timestamp() * 1000}
            override_window_start = None
            skip_step = False
            for channel, handle in handles.items():
                done, current_time, values = self._process_channel(
                    channel,
                    handle,
                    window_start,
                    window_end,
                    approximate_steps[channel],
                )
                if not done:
                    LOGGER.info(
                        f"Current file exhausted at time {current_time}. Preparing new file handle.",
                        identifier=channel,
                    )
                    new_handle, t, stop = self._prepare_file_handle(
                        current_time,
                        channel,
                        approximate_step=approximate_steps[channel],
                        previous_file_exhausted=True,
                    )
                    handles[channel] = new_handle
                    if stop:
                        should_stop = True
                        skip_step = True
                        break
                    if t != current_time:
                        LOGGER.warn(
                            "Adjusting window start to new file handle due to missing data."
                        )
                        override_window_start = t
                    else:
                        _, _, additional_values = self._process_channel(
                            channel,
                            new_handle,
                            current_time,
                            window_end,
                            approximate_steps[channel],
                        )
                        values += additional_values
                value = self._sampling_strategy.sample(values, target)
                if value is None:
                    if last_values is not None:
                        LOGGER.warn(
                            f"Could not sample data for channel {channel} at time {target} (no values). Using previous value."
                        )
                        data[channel] = last_values[channel]
                    else:
                        LOGGER.warn(
                            f"Could not sample data for channel {channel} at time {target} (no values). Skipping this step."
                        )
                        skip_step = True
                else:
                    data[channel] = value
            if not skip_step:
                yield data
                last_values = data
            window_start = override_window_start or window_end
            window_end = window_start + datetime.timedelta(seconds=self._resolution)

    def _prepare_file_handle(
        self,
        time: datetime.datetime,
        channel: str,
        approximate_step: Optional[datetime.timedelta] = None,
        previous_file_exhausted: bool = False,
    ) -> Tuple[Optional[BinaryIO], Optional[datetime.datetime], bool]:
        LOGGER.info(f"Preparing file handle at time {time}.", identifier=channel)
        f, missing, stop = self._open_file_handle(
            self._path, time, channel, forward=True
        )
        # find the first timestamp in the file
        if stop:
            return None, None, True
        t0, t_final = self._compute_file_span(f)
        if missing:
            return f, t0, False
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
                return f, t0, False

            LOGGER.info(f"Target too early. Switching files.", identifier=channel)
            f.close()
            f, missing, stop = self._open_file_handle(
                self._path, time - datetime.timedelta(days=1), channel, forward=False
            )
            if stop:
                return None, None, True
            t0, t_final = self._compute_file_span(f)
            LOGGER.info(f"New file spans {t0} - {t_final}.", identifier=channel)
            if not time <= t_final:
                LOGGER.warn(
                    "Switched files but still did not find target time stamp. Switching forward again"
                )
                f.close()
                f, _, _ = self._open_file_handle(
                    self._path, time, channel, forward=True
                )
                t0, t_final = self._compute_file_span(f)
                return f, t0, False
        elif time > t_final:
            LOGGER.info(f"Target too late. Switching files.", identifier=channel)
            f.close()
            f, missing, stop = self._open_file_handle(
                self._path, time + datetime.timedelta(days=1), channel, forward=True
            )
            if stop:
                return None, None, True
            t0, t_final = self._compute_file_span(f)
            LOGGER.info(f"New file spans {t0} - {t_final}.", identifier=channel)
            if not time >= t0:
                LOGGER.warn(
                    "Switched files but still did not find target time stamp. Using current time step as target instead."
                )
                t0, t_final = self._compute_file_span(f)
                return f, t0, False
        # seek to the correct time (last measurement prior to the target one)
        LOGGER.info("Seeking to target offset.", identifier=channel)
        f.seek(0)
        approximate_step = self._approximate_step(f)
        LOGGER.info(f"Approximate step: {approximate_step}.", identifier=channel)
        return (
            (
                self._jump_to(f, time, t0, approximate_step),
                time,
                False,
            )
            if not missing
            else (f, t0, False)
        )

    def _open_file_handle(
        self, path: Path, time: datetime.datetime, channel: str, forward: bool
    ) -> Tuple[Optional[BinaryIO], bool, bool]:
        file_path = self._make_file_path(path, time, channel)
        missing_file = False
        while not file_path.exists():
            missing_file = True
            LOGGER.warn(
                f"File {file_path} does not exist. This indicates missing data. Switching",
                identifier=channel,
            )
            time = (
                time + datetime.timedelta(days=1)
                if forward
                else time - datetime.timedelta(days=1)
            )
            file_path = self._make_file_path(path, time, channel)
            if time.year < 2017 or time.year > 2021:
                return None, False, True

        LOGGER.info(f"New file handle: {file_path}", identifier=channel)
        f = open(file_path, "rb")
        return f, missing_file, False

    def _compute_file_span(
        self, f: BinaryIO
    ) -> Tuple[datetime.datetime, datetime.datetime]:
        f.seek(0)
        t0 = self._read_timestamp(f)
        f.seek(-MEASUREMENT_SIZE_IN_BYTES, 2)
        t_final = self._read_timestamp(f)
        f.seek(0)
        return t0, t_final

    def _process_channel(
        self,
        channel: str,
        f: BinaryIO,
        start: datetime.datetime,
        end: datetime.datetime,
        approximate_step: datetime.timedelta,
    ):
        if not f.read(1):
            # There is no data left
            return False, start, []
        else:
            f.seek(-1, 1)

        current = self._read_measurement(f)
        current_time = self._make_datetime(current.time)
        f.seek(-MEASUREMENT_SIZE_IN_BYTES, 1)
        LOGGER.debug(
            f"Current time: {current_time} at position {f.tell()}", identifier=channel
        )
        # Only seek back if we are not at the beginning of the file. Necessary because of small gaps in the recording.
        if current_time > start:
            if f.tell() == 0:
                LOGGER.warn(
                    "File exhausted at start of file. This is probably caused by missing measurement data.",
                    identifier=channel,
                )
            else:
                self._find_linear(f, start, forward=False)

        generator = GeneratorWithReturnValue(
            self._read_until_buffered(f, end, approximate_step, current)
        )
        values = [m for m in generator]
        done = generator.value
        end_time = self._make_datetime(values[-1].time) if len(values) > 0 else start
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
        LOGGER.debug(
            f"Jumping {steps} steps (= {steps * MEASUREMENT_SIZE_IN_BYTES} bytes)."
        )
        f.seek(steps * MEASUREMENT_SIZE_IN_BYTES)
        if not f.read(1):
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
        self,
        f: BinaryIO,
        time: datetime.datetime,
        approx_step: datetime.timedelta,
        current: Optional[Measurement] = None,
    ) -> Generator[Measurement, Any, bool]:
        try:
            if not current:
                current = self._read_measurement(f)
                yield current
            if current.time / 1000 > time.timestamp():
                return True
            num_measurements = int(
                (time - self._make_datetime(int(current.time))) / approx_step
            )
            if num_measurements < 0:
                raise ValueError(
                    f"Negative number of measurements to read: {num_measurements}. Current time: {current.time}, target time: {time.timestamp()}."
                )
            if num_measurements == 0:
                return True
            buffer = f.read(num_measurements * MEASUREMENT_SIZE_IN_BYTES)
            offset = 0
            if len(buffer) == 0:
                return False
            while offset < len(buffer):
                current = self._read_measurement_from_buffer(
                    buffer[offset : offset + MEASUREMENT_SIZE_IN_BYTES]
                )
                offset += MEASUREMENT_SIZE_IN_BYTES
                yield current
                if current.time / 1000 >= time.timestamp():
                    f.seek(-len(buffer) + offset, 1)
                    return True
            return (yield from self._read_until_buffered(f, time, approx_step, current))
        except struct.error:
            return False

    @staticmethod
    def _make_file_path(path: Path, time: datetime.datetime, channel: str):
        return path / f"{time:%Y}" / f"{time:%m}" / f"{time:%d}" / f"{channel}.dat"

    def _read_timestamp(self, f: BinaryIO) -> datetime.datetime:
        return self._make_datetime(
            struct.unpack("<q", f.read(MEASUREMENT_SIZE_IN_BYTES)[:8])[0]
        )

    @staticmethod
    def _read_measurement_from_buffer(buffer: bytes) -> Measurement:
        time, value = struct.unpack("<qf", buffer)
        return Measurement(measurement=value, time=time)

    @staticmethod
    def _read_measurement(f: BinaryIO) -> Measurement:
        time, value = struct.unpack("<qf", f.read(MEASUREMENT_SIZE_IN_BYTES))
        return Measurement(measurement=value, time=time)

    def _approximate_step(self, f: BinaryIO) -> datetime.timedelta:
        current = f.tell()
        f.seek(0)
        t0 = self._read_timestamp(f)
        f.seek(-MEASUREMENT_SIZE_IN_BYTES, 2)
        t1 = self._read_timestamp(f)
        length = f.tell()
        f.seek(current)
        return (t1 - t0) / (length / MEASUREMENT_SIZE_IN_BYTES)

    @staticmethod
    def _make_datetime(timestamp: int, is_millis: bool = True) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(
            timestamp / (1000 if is_millis else 1), tz=datetime.timezone.utc
        )


class GrandStandAssembler:
    def __init__(self, path: Path):
        self._path = path

    def assemble(
        self, scenario: str, channels: List[str]
    ) -> Generator[Dict[str, float], Any, None]:
        path = self._make_path(scenario)
        with open(path, "r") as f:
            line, channel_order = self._parse_channel_order(f)
            if not channel_order:
                raise ValueError(f"Failed to parse metadata for scenario {scenario}.")
            yield self._annotate(self._parse_line(line), channel_order, 0)
            for i, line in enumerate(f):
                yield self._annotate(self._parse_line(line), channel_order, i + 1)

    @staticmethod
    def _annotate(
        data: List[float],
        channel_order: List[str],
        measurement_index: int,
        desired_channels: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        time_index = channel_order.index("time")
        data[time_index] = measurement_index
        return {
            channel: value
            for channel, value in zip(channel_order, data)
            if desired_channels is None or channel in desired_channels
        }

    @staticmethod
    def _parse_line(line: str) -> List[float]:
        return [float(x) for x in line.split("\t")]

    @staticmethod
    def _parse_channel_order(f: TextIO) -> Tuple[str, Optional[List[str]]]:
        order = None
        while (line := f.readline()).startswith('"'):
            if line.startswith('"Y Axis DOFS"'):
                order = ["time"] + [
                    "Joint " + x.strip().replace('"', "") for x in line.split("\t")[1:]
                ]
        return line, order

    def _make_path(self, scenario: str) -> Path:
        return self._path / f"{scenario}.txt"


class Z24Assembler:
    def __init__(
        self, path: Path, sampling_strategy: SamplingStrategy, resolution: float
    ):
        self._path = path
        self._parser = Z24UndamagedParser()
        self._sampling_strategy = sampling_strategy
        self._resolution = resolution

    @staticmethod
    def _make_environmental_data(
        data: ParsedZ24File, channels: List[str] | None
    ) -> Dict[str, float]:
        def _mean(values: List[Measurement]) -> float:
            return mean([x.measurement for x in values])

        pre = data.pre_measurement_environmental_data
        post = data.post_measurement_environmental_data
        return {
            channel: (
                _mean(pre[channel].measurements) - _mean(post[channel].measurements)
            )
            / 2
            for channel in pre.keys()
            if channels is None or channel in channels
        }

    @staticmethod
    def _find_start_time(data: ParsedZ24File) -> int:
        channel = list(data.acceleration_data.keys())[0]
        return min([x.time for x in data.acceleration_data[channel].measurements])

    def _make_acceleration_data(self, data: ParsedZ24File, channels: List[str] | None):
        def _find_until(
            values: List[Measurement], timestamp: float, offset: int = 0
        ) -> List[Measurement]:
            results = []
            for value in values[offset:]:
                if value.time <= timestamp:
                    results.append(value)
                else:
                    break
            return results

        sampled = {}
        for channel, values in data.acceleration_data.items():
            if channels is not None and channel not in channels:
                continue
            # No need to sample
            if self._resolution == 0:
                sampled[channel] = [m.measurement for m in values.measurements]
                continue
            # All in milliseconds
            end_time = (
                min([x.time for x in values.measurements]) + self._resolution * 1000
            )
            offset = 0
            sampled_values = []
            while offset < len(values.measurements):
                sample = _find_until(values.measurements, end_time, offset)
                target_time = datetime.datetime.fromtimestamp(
                    end_time / 1000 - self._resolution / 2
                )
                value = self._sampling_strategy.sample(
                    sample,
                    target_time,
                )
                if value is None:
                    LOGGER.warn(
                        f"Failed to sample value for channel {channel} at time {target_time}."
                    )
                    continue
                end_time += self._resolution * 1000
                sampled_values.append(value)
                offset += len(sample)
            sampled[channel] = sampled_values
        return sampled

    def _make_measurement_dict(
        self, data: ParsedZ24File, channels: List[str] | None
    ) -> Generator[Dict[str, float], Any, None]:
        start_time = self._find_start_time(data)
        environmental_data = self._make_environmental_data(data, channels)
        acceleration_data = self._make_acceleration_data(data, channels)
        lengths = list(set([len(x) for x in acceleration_data.values()]))
        if len(lengths) != 1:
            shortest = min(lengths)
            LOGGER.warn(
                f"Unequal lengths for acceleration data: {lengths}. Cutting to {shortest}."
            )
            acceleration_data = {
                channel: value[:shortest]
                for channel, value in acceleration_data.items()
            }
        for i in range(lengths[0]):
            sample_acceleration_data = {
                channel: value[i] for channel, value in acceleration_data.items()
            }
            # Start time + middle of current sample in milliseconds
            time = start_time + 1.5 * i * self._resolution * 1000
            yield {"time": time, **environmental_data, **sample_acceleration_data}

    def assemble(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        channels: List[str] | None = None,
    ) -> Generator[Dict[str, float] | EOF, Any, None]:
        for data in self._parser.parse(self._path, start_time, end_time):
            yield from self._make_measurement_dict(data, channels)
            yield EOF()


class Z24DamagedAssembler:
    def __init__(self, zip_directory: Path):
        self._zip_directory = zip_directory
        self._parser = Z24DamagedParser()

    def assemble(
        self, scenario: int, mode: Literal["avt", "fvt"]
    ) -> Generator[Dict[str, float], Any, None]:
        data = self._parser.parse(self._zip_directory, scenario, mode)
        lengths = list(
            set([len(x.measurements) for x in data.acceleration_data.values()])
        )
        if len(lengths) != 1:
            shortest = min(lengths)
            LOGGER.warn(
                f"Unequal lengths for acceleration data: {lengths}. Cutting to {shortest}."
            )
            data = {channel: value[:shortest] for channel, value in data.items()}
        for i in range(lengths[0]):
            step_data = {
                channel: value.measurements[i].measurement
                for channel, value in data.acceleration_data.items()
            }
            step_data["time"] = i
            yield step_data


class AssemblerWrapper:
    def __init__(
        self,
        mode: str,
        base_assembler: PBimAssembler
        | GrandStandAssembler
        | Z24Assembler
        | Z24DamagedAssembler,
    ):
        self._mode = mode
        self._base_assembler = base_assembler

    def assemble(
        self,
        start_time: Optional[datetime.datetime],
        end_time: Optional[datetime.datetime],
        scenario: Optional[str],
        channels: Optional[List[str]],
    ) -> Generator[Dict[str, float] | EOF, Any, None]:
        match self._mode:
            case "pbim":
                yield from self._base_assembler.assemble(start_time, end_time, channels)
            case "grandstand":
                yield from self._base_assembler.assemble(scenario, channels)
            case "z24":
                yield from self._base_assembler.assemble(start_time, end_time, channels)
            case _:
                raise ValueError(f"Unknown mode {self._mode}")
