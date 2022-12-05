import sys
from pathlib import Path

from parser.parser import PBimParser


def main():
    path = Path(sys.argv[1])
    parser = PBimParser()
    data = parser.parse(path)
    print(data)

if __name__ == "__main__":
    main()
