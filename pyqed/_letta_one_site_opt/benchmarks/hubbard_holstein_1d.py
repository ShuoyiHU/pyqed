"""Click-run 1D Hubbard-Holstein comparison using the CBE paper's Hamiltonian.

Local defaults: L=3, D=4, max_phonons=1 (d=8), at most 50 directional passes.
Adjust --length, --bond-dim, --max-phonons, --t, --U, --omega, --g and --mu.
The paper uses fixed N=L and S=0; this benchmark leaves both unconstrained.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed._letta_one_site_opt.benchmarks.condensed_cli import run_model_cli


def main(argv=None):
    return run_model_cli("hubbard_holstein", "1d", argv)


if __name__ == "__main__":
    main()
