import concurrent
import datetime
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

import click

from pbim_preprocessor.cli import CHANNELS_TO_PROCESS
from pbim_preprocessor.processor import Processor
from pbim_preprocessor.utils import LOGGER


def _parse_name(name: str) -> datetime.datetime:
    return datetime.datetime.strptime(Path(name).stem, "Job1_%Y_%m_%d_%H_%M_%S")


def _split(names: List[str], n_chunks: int) -> List[List[str]]:
    chunk_size = max(len(names) // n_chunks, 1)
    chunks = []
    current_chunk = []

    for i, name in enumerate(names):
        target_chunk = i // chunk_size
        if target_chunk > len(chunks):
            if _parse_name(name).day != _parse_name(names[i - 1]).day:
                chunks.append(current_chunk)
                current_chunk = []
        current_chunk.append(name)
    chunks.append(current_chunk)
    return chunks


def _process(
    zip_file_path: Path, output_base_path: Path, tmp_path: Path, names: List[str]
):
    processor = Processor(
        zip_file_path, output_base_path, tmp_path, names, CHANNELS_TO_PROCESS
    )
    processor.process()


@click.command()
@click.argument("zip_file_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_base_path", type=click.Path(path_type=Path, writable=True))
@click.option("--workers", default=8, help="Number of workers to use.")
@click.option(
    "--reset", default=False, is_flag=True, help="Reset the output directory."
)
@click.option(
    "--tmp",
    default=Path("/tmp/out"),
    type=click.Path(path_type=Path),
    help="Path to the temporary extraction directory.",
)
def process(zip_file_path: Path, output_base_path: Path, workers: int, reset: bool, tmp: Path):
    if output_base_path.exists() and reset:
        LOGGER.warn(f"Removing output directory '{output_base_path}'")
        shutil.rmtree(output_base_path)
    output_base_path.mkdir(exist_ok=True, parents=True)
    with zipfile.ZipFile(zip_file_path, "r") as zip_file:
        file_names = zip_file.namelist()
    names = sorted(list(set([Path(name).stem for name in file_names])), key=_parse_name)
    LOGGER.info(f"Processing {len(names)} measurement jobs using {workers} workers.")

    chunks = _split(names, workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_process, zip_file_path, output_base_path, tmp, chunk)
            for chunk in chunks
        ]
        concurrent.futures.wait(futures)
