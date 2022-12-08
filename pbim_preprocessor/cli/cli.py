import click

from pbim_preprocessor.cli.parse import parse
from pbim_preprocessor.cli.plot import plot
from pbim_preprocessor.cli.process import process


@click.group()
def cli():
    pass


cli.add_command(parse)
cli.add_command(plot)
cli.add_command(process)
