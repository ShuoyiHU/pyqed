"""Optional trusted, atomic, process-locked cache of compiled chemistry MPOs."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class ComponentReference:
    """Expand one local core only when a reference contraction requests it."""
    mpo: object
    site: int

    @property
    def component_shape(self):
        from .reduced_symmetry import _sector_irrep
        core = self.mpo.sites[self.site]
        left = sum(_sector_irrep(q).dim for q in core.qns[0])
        right = sum(_sector_irrep(q).dim for q in core.qns[2])
        d = self.mpo.physical_basis.dense_dim
        return left, right, d, d

    def as_dense(self):
        from .reduced_contraction import expand_reduced_mps_site
        d = self.mpo.physical_basis.dense_dim
        return np.einsum('apb,oip->aboi', expand_reduced_mps_site(self.mpo.sites[self.site]),
                         self.mpo.local_transform.reshape(d, d, -1))


@lru_cache(maxsize=1)
def _implementation_signature():
    """Invalidate compiled representations when their Python implementation changes."""
    package = Path(__file__).resolve().parents[1]
    files = set()
    for folder in ('_letta_one_site_opt', 'mps/nonabelian'):
        files.update((package/folder).glob('*.py'))
    files.update((package/'mps'/name) for name in ('su2.py', 'symmetry.py'))
    digest = hashlib.sha256(b'qc-native-mpo-cache-v1')
    for path in sorted(files):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _cache_key(problem, cutoff):
    digest = hashlib.sha256(_implementation_signature().encode())
    metadata = dict(cutoff=float(cutoff), ecore=float(problem.ecore),
                    order=list(problem.orbital_order), nelec=list(problem.nelec),
                    hubbard=os.environ.get('PYQED_HUBBARD_MPO', 'sparse'))
    digest.update(json.dumps(metadata, sort_keys=True).encode())
    for array in (problem.h1, problem.eri):
        a = np.ascontiguousarray(array)
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def cached_hamiltonian(problem, cutoff, directory):
    """One builder per key; other jobs wait and read the atomically saved result.

    Locking uses flock on the cluster's shared filesystem. No MPS state or D
    enters the cache: this is solely an operator representation. Solver and
    compiler tolerances are unchanged. Only trusted cache directories may be used.
    """
    import fcntl
    from .qchem_workflow import compact_hamiltonian
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = _cache_key(problem, cutoff)
    path = directory/(key+'.pkl')
    tick = perf_counter()
    with (directory/(key+'.lock')).open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            with path.open('rb') as stream:
                payload = pickle.load(stream)
            if payload['key'] != key:
                raise ValueError(f'Hamiltonian cache signature mismatch: {path}')
            print(f'[QC MPO cache] hit {key[:12]} ({perf_counter()-tick:.3f} s, including lock wait)', flush=True)
            return payload['hamiltonian']
        print(f'[QC MPO cache] building {key[:12]}', flush=True)
        ham = compact_hamiltonian(problem._build_su2_mpo(cutoff=cutoff),
                                  problem.symmetry('su2').physical_basis)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, prefix=key+'.', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                pickle.dump(dict(key=key, hamiltonian=ham), stream, pickle.HIGHEST_PROTOCOL)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f'[QC MPO cache] saved {path.stat().st_size} bytes ({perf_counter()-tick:.3f} s)', flush=True)
        return ham
