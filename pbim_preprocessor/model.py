from dataclasses import dataclass
from typing import Optional, List
import datetime

from dataclasses_json import dataclass_json


@dataclass(frozen=True)
class GlobalHeader:
    date: Optional[datetime.datetime] = None
    time: Optional[datetime.datetime] = None


@dataclass_json
@dataclass(frozen=True)
class ChannelHeader:
    name: Optional[str] = None
    unit: Optional[str] = None
    encoding: Optional[str] = None
    dtype: Optional[str] = None
    bit_mask: Optional[int] = None
    channel_type: Optional[str] = None
    data_offset: Optional[int] = None
    measurements: Optional[int] = None


@dataclass
class Measurement:
    measurement: float
    # Epoch in milliseconds
    time: Optional[int] = None


@dataclass
class ParsedChannel:
    channel_header: ChannelHeader
    measurements: List[Measurement]
