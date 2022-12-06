from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json


@dataclass(frozen=True)
class GlobalHeader:
    date: Optional[str] = None
    time: Optional[str] = None


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
