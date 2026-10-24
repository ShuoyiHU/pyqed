"""Small synthetic chemistry preparation benchmark; no solver or dense FCI.

Run in separate processes for meaningful peak RSS comparisons. Disk caching
is explicitly disabled unless --cache-dir is supplied. Set BLAS threads to 1.
"""
import argparse
import json
import resource
import sys
from time import perf_counter
import numpy as np
from pyqed._letta_one_site_opt.qchem import ElectronicProblem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orbitals', type=int, default=4)
    parser.add_argument('--cache-dir')
    args = parser.parse_args()
    n = args.orbitals
    rng = np.random.default_rng(431)
    h = rng.normal(size=(n, n)); h = (h+h.T)/2
    f = rng.normal(size=(3, n, n)); f = (f+f.transpose(0, 2, 1))/2
    eri = np.einsum('xpq,xrs->pqrs', f, f)
    problem = ElectronicProblem(h, eri, (n//2, n//2), .37)
    tick = perf_counter()
    ham = problem.su2_mpo(cache_dir=args.cache_dir or False)
    build = perf_counter()-tick
    tick = perf_counter()
    native = ham.native_mpo(problem.symmetry('su2').physical_basis)
    compile_time = perf_counter()-tick
    shapes = [getattr(w, 'component_shape', None) or
              (w.left_dim, w.right_dim, 4, 4) for w in ham.canonical_factors]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        rss *= 1024
    print(json.dumps(dict(orbitals=n, cache_enabled=bool(args.cache_dir),
        su2_mpo_seconds=build, native_mpo_seconds=compile_time,
        total_preparation_seconds=build+compile_time, peak_process_rss_bytes=rss,
        component_shapes=shapes,
        component_dense_elements=sum(int(np.prod(s)) for s in shapes),
        native_elements=sum(v.size for s in native.sites for v in s.data.values()),
        reconstruction_residual=native.relative_reconstruction_error), indent=2))


if __name__ == '__main__':
    main()
