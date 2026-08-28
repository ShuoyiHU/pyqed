"""Click-to-run LETTA benchmark comparing dense, U(1), and SU(2) states.

Open this file in an IDE and click its Run button.  Adjust the constants below
to make the benchmark faster or to test a larger system.  The benchmark checks
energy agreement before its timing and memory reductions should be interpreted.

Use Python 3.9 or newer with the project's numerical dependencies installed.
"""

import sys
from pathlib import Path
from types import ModuleType


# User-editable benchmark settings.
NSITES = 6
SOLVER = "both"  # "one-site", "two-site", or "both"
MULTIPLETS_PER_SECTOR = 3
MAX_SWEEPS = 8
TOLERANCE = 1.0e-9
REPEATS = 1
SEED = 7


def _benchmark_main():
    """Import LETTA without running unrelated top-level pyqed imports."""

    if "pyqed" not in sys.modules:
        package = ModuleType("pyqed")
        package.__path__ = [str(Path(__file__).resolve().parent / "pyqed")]
        package.__package__ = "pyqed"
        sys.modules["pyqed"] = package
    from pyqed._letta_two_site_opt.benchmarks.heisenberg_symmetry import main

    return main


def main():
    """Run the symmetry benchmark and return its full result dictionary."""

    return _benchmark_main()(
        [
            "--nsites",
            str(NSITES),
            "--solver",
            SOLVER,
            "--multiplets-per-sector",
            str(MULTIPLETS_PER_SECTOR),
            "--max-sweeps",
            str(MAX_SWEEPS),
            "--tolerance",
            str(TOLERANCE),
            "--repeats",
            str(REPEATS),
            "--seed",
            str(SEED),
        ]
    )


if __name__ == "__main__":
    main()
