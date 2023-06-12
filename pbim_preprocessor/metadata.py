import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Literal

from dataclasses_json import dataclass_json

from pbim_preprocessor.statistic import ChannelStatistics


@dataclass_json
@dataclass
class DatasetMetadata:
    channel_order: List[str]
    start_time: Optional[int]
    end_time: Optional[int]
    measurement_size_in_bytes: int
    resolution: Optional[int]
    length: int
    statistics: Dict[str, ChannelStatistics]
    time_byte_size: int
    time_unit: Literal["milliseconds", "microseconds"] = "milliseconds"

    def is_time_in_milliseconds(self) -> bool:
        return self.time_unit == "milliseconds"

    def is_time_in_microseconds(self) -> bool:
        return self.time_unit == "microseconds"


def _write_metadata_file(path: Path, metadata: DatasetMetadata):
    metadata_path = path.parent / f"{path.stem}.metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(
            metadata.to_dict(),
            f,
            indent=4,
        )
