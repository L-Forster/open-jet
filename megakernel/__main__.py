"""Run OpenJet Engine with ``python -m megakernel`` or the ``openjet-engine`` command."""

import sys
from pathlib import Path


def main():
    # the server's modules import each other by bare name, as they do when server/mk_server.py runs as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent / "server"))
    from mk_server import main as serve

    serve()


if __name__ == "__main__":
    main()
