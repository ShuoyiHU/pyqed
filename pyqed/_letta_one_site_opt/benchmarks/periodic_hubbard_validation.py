"""Independent full fixed-N wavefunction checks, never an optimization path."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from pyqed._letta_one_site_opt.periodic import PeriodicState
from .periodic_hubbard import save


def half_filled_basis(length):
    bits = np.sort(np.fromiter((sum(1 << k for k in occupied)
                               for occupied in itertools.combinations(range(2*length), length)),
                              dtype=np.uint32))
    configs = ((bits[:, None] >> (2*np.arange(length, dtype=np.uint32))) & 3).astype(np.int8)
    return bits, configs


def physical_check(state, basis=None, hopping=1., interaction=4.):
    """Occupation-bit Hubbard action on all C(2L,L) configurations, in batches."""
    length = state.nsites
    bits, configs = half_filled_basis(length) if basis is None else basis
    v = np.concatenate([state.amplitudes(configs[i:i+2048]) for i in range(0,len(bits),2048)])
    norm = float(np.vdot(v,v).real)
    hv = interaction*np.sum(configs==3,axis=1)*v
    for i in range(length):
        j=(i+1)%length
        for spin in (0,1):
            for target, source in ((2*i+spin,2*j+spin),(2*j+spin,2*i+spin)):
                selected=np.flatnonzero(((bits >> source)&1) & (1-((bits >> target)&1)))
                middle=bits[selected] ^ np.uint32(1 << source)
                destination=np.searchsorted(bits,middle | np.uint32(1 << target))
                parity=(np.bitwise_count(bits[selected] & np.uint32((1 << source)-1))
                        +np.bitwise_count(middle & np.uint32((1 << target)-1)))%2
                signs=1.-2.*parity
                hv[destination] += -hopping*signs*v[selected]
    energy=float((np.vdot(v,hv)/norm).real)
    variance=float(np.vdot(hv-energy*v,hv-energy*v).real/norm)
    return dict(physical_energy=energy,physical_norm=norm,energy_variance=variance,
                basis_dimension=len(bits),mean_nup=float(np.sum(np.abs(v)**2*np.sum(configs&1,axis=1))/norm))


def validate_results(output):
    report=[]
    for length in (5,10):
        basis=half_filled_basis(length)
        for path in sorted(output.glob(f'L{length}_D*_seed*.json')):
            r=json.loads(path.read_text())
            if r.get('status')!='finished':continue
            with np.load(path.with_suffix('.npz')) as data:
                state=PeriodicState([data[f'tensor_{i}'] for i in range(length)],r['method'],np.array(r['charges']))
            checked=physical_check(state,basis)
            checked.update(case=path.stem,contracted_energy=r['final_energy'],
                           contraction_error=abs(checked['physical_energy']-r['final_energy']))
            if checked['contraction_error'] > 1e-8:
                raise AssertionError(checked)
            report.append(checked);save(output/'physical_validation.json',report)
            print(json.dumps(checked),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    validate_results(p.parse_args().output)
