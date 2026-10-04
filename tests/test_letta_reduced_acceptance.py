"""Variational acceptance must survive ill-conditioned cached environments."""
import json
from pathlib import Path
import numpy as np
from pyqed._letta_one_site_opt import ReducedLatticeLETTA, LETTADMROptions, letta_dmrg
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.reduced_symmetry import SpinChargeSector, SU2Irrep
from pyqed._letta_one_site_opt.reduced_solver import _energy


def molecular_roundoff_case():
    with np.load(Path(__file__).parent/'data/feco5_cas8_nn_roundoff.npz') as z:
        p=ElectronicProblem(z['h1'],z['eri'],tuple(z['nelec']),float(z['ecore']))
        meta=json.loads(str(z['metadata']))
        def sector(q):return SpinChargeSector(q[0],SU2Irrep(q[1]))
        cores=[{tuple(map(sector,key)):z[name].copy() for name,key in core} for core in meta['cores']]
    state=ReducedLatticeLETTA((1,p.norb),p.symmetry('su2'),cores,
        bond_sectors=[tuple(map(sector,b)) for b in meta['bonds']],
        neighborhoods=tuple(map(tuple,meta['neighborhoods'])),normalize=False)
    return p,state


def check_sweep(state,h):
    before=_energy(state,h,stable=True)
    result=letta_dmrg(h,state=state,options=LETTADMROptions(max_sweeps=1,
        tolerance=1e-10,eigensolver_tolerance=1e-11,energy_increase_tolerance=1e-11,
        gauge_mode='frontier',dense_solver_threshold=32,start_direction='lr'))
    assert result.energy <= before+1e-9, (before,result.energy)
    np.testing.assert_allclose(result.energy,_energy(result.state,h,stable=True),atol=1e-10,rtol=0)
    return result


def test_cached_reduced_sweep_acceptance_uses_stable_energy(monkeypatch):
    p,state=molecular_roundoff_case()
    def forbidden(*args,**kwargs):raise AssertionError('active solver expanded a global determinant vector')
    monkeypatch.setattr(ReducedLatticeLETTA,'state_vector',forbidden)
    check_sweep(state,p.su2_mpo())
