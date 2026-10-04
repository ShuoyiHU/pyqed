#!/usr/bin/env python3
"""3x6 LETTA comparisons using the same frozen solver as the October 2 runs."""
import importlib.util
from pathlib import Path
import sys

CASES = tuple((model, (6, 3), 3 if model == 'fermi_hubbard' else 4)
              for model in ('ising', 'heisenberg', 'bose_hubbard', 'fermi_hubbard'))


def driver(root):
    path = root/'source/pyqed/_letta_one_site_opt/benchmarks/2D/cluster_sept27/run_compression.py'
    spec = importlib.util.spec_from_file_location('rectangle_compression_driver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CASES = CASES
    return module


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(__file__).resolve().parent
    if '--root' in argv:
        root = Path(argv[argv.index('--root')+1]).resolve()
    elif argv and argv[0] != 'collect':
        argv += ['--root', str(root)]
    if argv and argv[0] == 'plan':
        if '--algorithms' not in argv:
            argv += ['--algorithms', 'one-site', 'cbe', 'two-site']
        if '--profiles' not in argv:
            argv += ['--profiles', 'als4-lsmr400', 'als40-lsmr400']
    return driver(root).main(argv)


if __name__ == '__main__':
    sys.exit(main())
