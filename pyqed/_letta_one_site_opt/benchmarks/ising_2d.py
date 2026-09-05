"""Click-run 2D transverse-field Ising benchmark."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed._letta_one_site_opt.benchmarks.condensed_cli import run_model_cli


def main(argv=None):
    return run_model_cli("ising", "2d", argv)


if __name__ == "__main__":
    main()
