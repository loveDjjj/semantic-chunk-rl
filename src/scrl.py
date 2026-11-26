"""Expose a flat source checkout as the public scrl package on PYTHONPATH."""
from pathlib import Path

if __name__ == "__main__":
    from scrl.cli import main

    raise SystemExit(main())
else:
    __path__ = [str(Path(__file__).resolve().parent)]
    __package__ = __name__
    __spec__.submodule_search_locations = __path__
