import datetime
import struct
from pathlib import Path
from typing import List, BinaryIO, Dict, TextIO, Callable, Tuple

import scipy

from pbim_preprocessor.model import (
    PBimChannelHeader,
    Measurement,
    ParsedPBimChannel,
    ParsedZ24Channel,
    Z24ChannelHeader,
)


class PBimDataParser:
    def parse(
        self, directory: Path, name: str, channel: PBimChannelHeader
    ) -> ParsedPBimChannel:
        encoding = self._infer_encoding(channel)
        actual_offset = self._compute_actual_offset(channel)
        with open(directory / f"{name}.R32", "rb") as f:
            f.seek(actual_offset)
            return ParsedPBimChannel(
                channel_header=channel,
                measurements=[
                    self._parse_measurement(f, encoding)
                    for _ in range(channel.measurements)
                ],
            )

    def parse_all(
        self, directory: Path, name: str, channel_headers: List[PBimChannelHeader]
    ) -> Dict[str, ParsedPBimChannel]:
        return {
            channel_header.name: self.parse(directory, name, channel_header)
            for channel_header in channel_headers
        }

    def _compute_actual_offset(self, channel: PBimChannelHeader) -> int:
        encoding = self._infer_encoding(channel)
        size = struct.calcsize(encoding)
        # offset is given in "records" instead of bytes initially
        return (channel.data_offset - 1) * size

    @staticmethod
    def _parse_measurement(f: BinaryIO, encoding: str) -> Measurement:
        # data is encoded in little-endian
        data = f.read(struct.calcsize(encoding))
        value = struct.unpack(encoding, data)[0]
        return Measurement(measurement=value)

    @staticmethod
    def _infer_encoding(channel: PBimChannelHeader) -> str:
        match channel.dtype.lower():
            case "int32":
                return "<i"
            case "int16":
                return "<H"
            case "real32":
                return "<f"
            case "real64":
                return "<d"
            case _:
                raise ValueError(f"Unknown data type {channel.dtype}")


class Z24AccelerationDataParser:
    @staticmethod
    def _parse_footer(f: TextIO) -> Tuple[datetime.datetime, datetime.datetime]:
        def _find(prefix: str):
            while not (line := f.readline()).startswith(prefix):
                pass
            return line

        start = datetime.datetime.strptime(
            _find(f"Segment #1").split(":", maxsplit=1)[1].strip(),
            "%a %b %d %H:%M:%S %Y",
        )
        _find("Segment #8")
        stop = datetime.datetime.strptime(
            _find(" Stop").split(":", maxsplit=1)[1].strip(), "%a %b %d %H:%M:%S %Y"
        )
        return start, stop

    @staticmethod
    def _annotate_times(
        data: List[float], start: datetime.datetime, end: datetime.datetime
    ) -> List[Measurement]:
        measurements = []
        span = (end - start).total_seconds()
        for i, measurement in enumerate(data):
            time = start + datetime.timedelta(seconds=span * i / len(data))
            measurements.append(
                Measurement(measurement=measurement, time=int(time.timestamp() * 1000))
            )
        return measurements

    def parse(self, f: TextIO) -> ParsedZ24Channel:
        name = str(f.readline().strip())
        num_samples = int(f.readline().strip())
        frequency = float(f.readline().strip())
        data = []
        for i in range(num_samples):
            measurement = float(f.readline().strip())
            data.append(measurement)

        start, stop = self._parse_footer(f)
        return ParsedZ24Channel(
            channel_header=Z24ChannelHeader(
                name=name,
                frequency=frequency,
                num_samples=num_samples,
            ),
            measurements=self._annotate_times(data, start, stop),
        )


class Z24EnvironmentalDataParser:
    @staticmethod
    def _parser_header(f: TextIO) -> List[str]:
        headers = [w for w in f.readline().split()]
        return headers[::2]

    @staticmethod
    def _convert(data: List[Measurement], name: str) -> List[Measurement]:
        def _map(m: Measurement, f: Callable[[float], float]) -> Measurement:
            return Measurement(measurement=f(m.measurement), time=m.time)

        match name:
            case "WS":
                return [_map(d, lambda x: x * 5) for d in data]
            case "WD":
                return [_map(d, lambda x: x * 36) for d in data]
            case "AT":
                return [_map(d, lambda x: x * 100 - 40) for d in data]
            case "R":
                return [_map(d, lambda x: 1 if x >= 3 else 0) for d in data]
            case "H":
                return [_map(d, lambda x: x * 10) for d in data]
            case "TE":
                return [_map(d, lambda x: x * 5) for d in data]
            case _:
                return data

    @staticmethod
    def _parse_footer(f: TextIO) -> Tuple[datetime.datetime, datetime.datetime]:
        def _find(prefix: str):
            while not (line := f.readline()).startswith(prefix):
                pass
            return line

        footer = _find("EnvScan started")
        start = datetime.datetime.strptime(
            footer.split(":", maxsplit=1)[1].strip(), "%a %b %d %H:%M:%S %Y"
        )
        footer = _find("stopped")
        end = datetime.datetime.strptime(
            footer.split(":", maxsplit=1)[1].strip(), "%a %b %d %H:%M:%S %Y"
        )
        return start, end

    @staticmethod
    def _annotate_times(
        data: List[float], times: Tuple[datetime.datetime, datetime.datetime]
    ) -> List[Measurement]:
        measurements = []
        start, end = times
        for i, measurement in enumerate(data):
            time = start + datetime.timedelta(
                seconds=i / len(data) * (end - start).total_seconds()
            )
            measurements.append(
                Measurement(measurement=measurement, time=int(time.timestamp()))
            )
        return measurements

    def parse(self, f: TextIO) -> Dict[str, ParsedZ24Channel]:
        header = self._parser_header(f)
        data = {}
        for line in f:
            if line.strip() == "":
                break
            line_data = [float(w) for w in line.strip().split()]
            for i in range(len(header)):
                if header[i] not in data:
                    data[header[i]] = []
                data[header[i]].append(line_data[i])

        times = self._parse_footer(f)
        data = {k: self._annotate_times(v, times) for k, v in data.items()}

        return {
            header[i]: ParsedZ24Channel(
                channel_header=Z24ChannelHeader(
                    name=header[i],
                    frequency=None,
                    num_samples=len(data[header[i]]),
                ),
                measurements=self._convert(data[header[i]], header[i]),
            )
            for i in range(len(header))
        }


class Z24PDTAccelerationParser:
    def __init__(self):
        pass

    def parse(self, path: Path) -> Dict[str, ParsedZ24Channel]:
        data = scipy.io.loadmat(str(path))
        labels = data["labelshulp"].tolist()
        measurements = data["data"]
        parsed = {}
        for i, label in enumerate(labels):
            header = Z24ChannelHeader(
                name=label, num_samples=measurements.shape[0], frequency=100
            )
            channel_measurements = [
                Measurement(measurement=m, time=j)
                for j, m in enumerate(measurements[:, i])
            ]
            parsed[label] = ParsedZ24Channel(
                channel_header=header, measurements=channel_measurements
            )
        return parsed
