import struct
from pathlib import Path
from typing import List, BinaryIO, Dict

from pbim_preprocessor.model import ChannelHeader, Measurement


class PBimDataParser:
    def parse(
        self, directory: Path, name: str, channel: ChannelHeader
    ) -> List[Measurement]:
        encoding = self._infer_encoding(channel)
        actual_offset = self._compute_actual_offset(channel)
        with open(directory / f"{name}.r32", "rb") as f:
            f.seek(actual_offset)
            return [
                self._parse_measurement(f, encoding)
                for _ in range(channel.measurements)
            ]

    def parse_all(
        self, directory: Path, name: str, channel_headers: List[ChannelHeader]
    ) -> Dict[ChannelHeader, List[Measurement]]:
        return {
            channel_header: self.parse(directory, name, channel_header)
            for channel_header in channel_headers
        }

    def _compute_actual_offset(self, channel: ChannelHeader) -> int:
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
    def _infer_encoding(channel: ChannelHeader) -> str:
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
