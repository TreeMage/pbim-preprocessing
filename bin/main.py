import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from pbim_preprocessor.cli.cli import cli

if __name__ == "__main__":
    cli()
