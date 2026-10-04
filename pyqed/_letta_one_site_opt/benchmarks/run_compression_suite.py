"""Run each reference benchmark in a fresh, single-thread local process."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from .metric_compression_solvers import METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sweeps', type=int, default=100)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               MPLCONFIGDIR='/private/tmp/letta-mpl-cache', PYTHONPATH='.',
               PYTHONUNBUFFERED='1')
    failures = []
    for shape, bond in [('2,2', 2), ('2,2', 3), ('2,2', 4), ('3,2', 3)]:
        for method in ('one',)+METHODS:
            name = f"bose{shape.replace(',', '')}_d{bond}_{method}"
            if (output/(name+'.json')).exists():
                continue
            print('START', name, flush=True)
            with (output/(name+'.log')).open('w') as log:
                result = subprocess.run([
                    sys.executable, '-m',
                    'pyqed._letta_one_site_opt.benchmarks.compression_accuracy', 'run',
                    '--shape', shape, '--bond', str(bond), '--method', method,
                    '--sweeps', str(args.sweeps), '--output', str(output)],
                    env=env, stdout=log, stderr=subprocess.STDOUT)
            print('FINISH', name, 'exit', result.returncode, flush=True)
            if result.returncode:
                failures.append(name)
    if failures:
        raise RuntimeError(f'Failed runs: {failures}')


if __name__ == '__main__':
    main()
