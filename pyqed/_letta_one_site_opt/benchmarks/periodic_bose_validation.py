"""Independent fixed-number occupation reference; never used by the optimizer."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from pyqed._letta_one_site_opt.periodic import PeriodicState


def sector_hamiltonian(length, hopping=1., interaction=4., max_occupancy=2):
    configs = np.array([s for s in itertools.product(range(max_occupancy+1), repeat=length)
                        if sum(s) == length], dtype=int)
    lookup = {tuple(s): i for i, s in enumerate(configs)}
    rows, cols, values = [], [], []
    for col, occupations in enumerate(configs):
        rows.append(col); cols.append(col)
        values.append(.5*interaction*np.sum(occupations*(occupations-1)))
        for i in range(length):
            j = (i+1) % length
            for target, source in ((i,j),(j,i)):
                if occupations[source] == 0 or occupations[target] == max_occupancy:
                    continue
                moved = occupations.copy()
                moved[target] += 1
                moved[source] -= 1
                rows.append(lookup[tuple(moved)]); cols.append(col)
                values.append(-hopping*np.sqrt(occupations[source]*(occupations[target]+1)))
    h = sparse.coo_matrix((values,(rows,cols)),shape=(len(configs),len(configs))).tocsr()
    return configs, h


def sector_reference(length, hopping=1., interaction=4., max_occupancy=2):
    configs, h = sector_hamiltonian(length, hopping, interaction, max_occupancy)
    e,v = eigsh(h,k=1,which='SA',tol=1e-12,
                v0=np.random.default_rng(8675309).normal(size=len(configs)))
    return dict(length=length, particles=length, max_occupancy=max_occupancy,
                energy=float(e[0]), residual=float(np.linalg.norm(h@v[:,0]-e[0]*v[:,0])),
                sector_dimension=len(configs))


def validate_results(output):
    from .periodic_hubbard import save
    results, references = [], {}
    for path in sorted(output.rglob('L*_seed*.json')):
        report = json.loads(path.read_text())
        if report.get('status') != 'finished':
            continue
        length = report['length']
        cutoff = report['model']['max_occupancy']
        key = (length, cutoff)
        if key not in references:
            references[key] = sector_hamiltonian(length, max_occupancy=cutoff)
        configs, h = references[key]
        with np.load(path.with_suffix('.npz')) as data:
            state = PeriodicState([data[f'tensor_{i}'] for i in range(length)],
                                  report['method'], np.array(report['charges']),
                                  tuple(report['particle_numbers']))
        v = state.amplitudes(configs)
        norm = np.vdot(v,v).real
        hv = h@v
        energy = float((np.vdot(v,hv)/norm).real)
        error = abs(energy-report['final_energy'])
        if error > 1e-8:
            raise AssertionError(f'{path}: energy mismatch {error}')
        results.append(dict(file=str(path.relative_to(output)), physical_energy=energy,
                            energy_discrepancy=error, physical_norm=float(norm),
                            energy_variance=float(np.vdot(hv-energy*v,hv-energy*v).real/norm)))
    save(output/'physical_validation.json',results)
    print(json.dumps(results,indent=2))
    return results


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    validate_results(p.parse_args().output)
