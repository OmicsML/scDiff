__all__ = [
    "HOMEDIR",
    "DATADIR",
    "RUNTIMEDIR",
]

from pathlib import Path

from scdiff.utils.misc import ensure_dir

HOMEDIR = Path(__file__).resolve().parents[1]
DATADIR = Path(HOMEDIR / "data")
RUNTIMEDIR = ensure_dir(HOMEDIR / "runtime")
RESULTSDIR = ensure_dir(HOMEDIR / "results")
