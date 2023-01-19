import click

from pbim_preprocessor.cli.assemble import assemble
from pbim_preprocessor.cli.generate import generate
from pbim_preprocessor.cli.merge import merge
from pbim_preprocessor.cli.parse import parse
from pbim_preprocessor.cli.plot import plot
from pbim_preprocessor.cli.process import process


@click.group()
def cli():
    pass


cli.add_command(parse)
cli.add_command(plot)
cli.add_command(process)
cli.add_command(assemble)
cli.add_command(generate)
cli.add_command(merge)
