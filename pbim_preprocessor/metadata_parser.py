import dataclasses
import datetime
import re
from enum import Enum
from pathlib import Path
from typing import Tuple, List

from pbim_preprocessor.model import GlobalHeader, PBimChannelHeader

HEADER_MAGIC = "DIAEXTENDED"
ENCODING = "cp1252"


class PBimMetadataParserState(Enum):
    NONE = 0
    GLOBAL_HEADER = 1
    CHANNEL_HEADER = 2


class MetadataParserError(Exception):
    def __init__(self, state: PBimMetadataParserState, line: str, line_number: int):
        self._state = state
        self._line = line
        self.line = line_number

    def __repr__(self):
        return f"ParserError(state={self._state}, line={self._line}, line_number={self.line})"


class PBimMetadataParser:
    def __init__(self):
        self._state = PBimMetadataParserState.NONE
        self._position = 0
        self._current_global_header = None
        self._current_channel_header = None
        self._channel_headers = []

    def parse(
        self, directory: Path, name: str
    ) -> Tuple[GlobalHeader, List[PBimChannelHeader]]:
        with open(directory / f"{name}.DAT", "r", encoding=ENCODING) as f:
            for line in f.readlines():
                self._parse_line(line.strip())
        return self._current_global_header, self._channel_headers

    def reset(self):
        self._state = PBimMetadataParserState.NONE
        self._position = 0
        self._current_global_header = None
        self._current_channel_header = None
        self._channel_headers = []

    def _parse_line(self, line: str):
        if line == HEADER_MAGIC:
            return
        elif line.startswith("#"):
            self._parse_command(line)
        elif re.match("^\\d+,.*$", line):
            self._parse_data(line)
        else:
            print(line)
            raise MetadataParserError(
                state=self._state, line=line, line_number=self._position
            )
        self._position += 1

    def _parse_command(self, line: str):
        match line:
            case "#BEGINGLOBALHEADER":
                if self._current_global_header:
                    raise MetadataParserError(
                        state=self._state, line=line, line_number=self._position
                    )
                self._current_global_header = GlobalHeader()
                self._state = PBimMetadataParserState.GLOBAL_HEADER
            case "#ENDGLOBALHEADER":
                self._state = PBimMetadataParserState.NONE
            case "#BEGINCHANNELHEADER":
                self._state = PBimMetadataParserState.CHANNEL_HEADER
                self._current_channel_header = PBimChannelHeader()
            case "#ENDCHANNELHEADER":
                if self._current_channel_header:
                    self._channel_headers.append(self._current_channel_header)
                    self._current_channel_header = None
                self._state = PBimMetadataParserState.NONE
            case _:
                raise MetadataParserError(
                    state=self._state, line=line, line_number=self._position
                )

    def _parse_data(self, line: str):
        match self._state:
            case PBimMetadataParserState.GLOBAL_HEADER:
                self._parse_global_header_line(line)
            case PBimMetadataParserState.CHANNEL_HEADER:
                self._parse_channel_header_line(line)
            case _:
                raise MetadataParserError(
                    state=self._state, line=line, line_number=self._position
                )

    def _parse_global_header_line(self, line: str):
        command, value = line.split(",", maxsplit=1)
        match int(command):
            case 104:
                self._current_global_header = dataclasses.replace(
                    self._current_global_header,
                    date=datetime.datetime.strptime(value, "%m-%d-%Y"),
                )
            case 105:
                self._current_global_header = dataclasses.replace(
                    self._current_global_header,
                    time=datetime.datetime.strptime(value, "%H:%M:%S"),
                )
            case _:
                return

    def _parse_channel_header_line(self, line: str):
        command, value = line.split(",", maxsplit=1)
        match int(command):
            case 200:
                self._current_channel_header = dataclasses.replace(
                    self._current_channel_header, name=value
                )
            case 202:
                self._current_channel_header = dataclasses.replace(
                    self._current_channel_header, unit=value
                )
            case 210:
                self._current_channel_header = dataclasses.replace(
                    self._current_channel_header, channel_type=value
                )
            case 213:
                self._current_channel_header = dataclasses.replace(
                    self._current_channel_header, encoding=value
                )
            case 214:
                self._current_channel_header = dataclasses.replace(
                    self._current_channel_header, dtype=value
                )
            case 215:
                self._current_channel_header = dataclasses.replace(
                    self._current_channel_header, bit_mask=int(value)
                )
            case 220:
                self._current_channel_header = dataclasses.replace(
                    self._current_channel_header, measurements=int(value)
                )
            case 221:
                self._current_channel_header = dataclasses.replace(
                    self._current_channel_header, data_offset=int(value)
                )
            case _:
                return
