"""QR local coordinates against independent physical-state references."""
from pathlib import Path
import pickle
from types import SimpleNamespace
import numpy as np
import pytest
from pyqed._letta_one_site_opt import ReducedLatticeLETTA, LETTADMROptions
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.reduced_frontier import ReducedFrontier
from pyqed._letta_one_site_opt.reduced_coordinates import ReducedLocalCoordinates
from pyqed._letta_one_site_opt.reduced_solver import _energy, _davidson_in_metric_coordinates

@pytest.mark.parametrize('nelec',[(1,1),(2,1),(2,0)])
def test_coordinates_preserve_norm_energy_and_adjoint(nelec):
    h=np.diag([.1,.2,.3]);h[0,1]=h[1,0]=-.7;h[0,2]=h[2,0]=.2
    g=np.zeros((3,)*4)
    for i in range(3):g[i,i,i,i]=4.
    p=ElectronicProblem(h,g,nelec);mpo=p.su2_mpo()
    s=ReducedLatticeLETTA.random((1,3),symmetry=p.symmetry('su2'),multiplets_per_sector=2,real=False,seed=43,neighborhoods=((0,1,2),(1,2),(2,)))
    f=ReducedFrontier.from_state(s);sites=f.to_mps(s)
    physical=s.state_vector();norm=(s.target_two_j+1)*np.vdot(physical,physical).real
    expected=_energy(s,mpo,stable=True)
    rng=np.random.default_rng(16)
    for site in range(3):
        c=ReducedLocalCoordinates(sites,f.site_embedding(s,site),site)
        c.set_hamiltonian(mpo.native_mpo(s.physical_basis));c.prepare(1e-12)
        x=c.embedding.pack_source(s.tensors[site])
        assert c.norm(x)==pytest.approx(norm,rel=3e-12,abs=1e-12)
        assert c.energy(x)==pytest.approx(expected,abs=2e-11)
        z=rng.normal(size=c.rank)+1j*rng.normal(size=c.rank)
        np.testing.assert_allclose(c.adjoint(c.metric(c.expand(z))),z,atol=2e-10,rtol=2e-10)
        np.testing.assert_allclose(c.orthogonal_adjoint(c.orthogonal_expand(z)),z,atol=2e-11,rtol=2e-11)
        np.testing.assert_allclose(c.apply(z),c.adjoint(c.source_hamiltonian(c.expand(z))),atol=2e-10,rtol=2e-10)


def test_captured_ladder_null_direction_and_local_root():
    # Trusted regression fixture: bg-compatible seed-4 2x4 Hubbard ladder,
    # D=20, first forward sweep before site 4, original revision 48f1584.
    # Includes only the reduced state and one false Gram-eigenvector (~65 KiB).
    with (Path(__file__).parent/'data/letta_overlap_2x4_d20.pkl').open('rb') as stream:
        fixture=pickle.load(stream)
    s=fixture['state'];site=fixture['site'];ghost=fixture['ghost']
    f=ReducedFrontier.from_state(s)
    c=ReducedLocalCoordinates(f.to_mps(s),f.site_embedding(s,site),site)
    c.prepare(1e-12)
    assert c.rank==318
    assert c.norm(ghost)<1e-11  # The old Gram calculation reported norm ~= 1.
    original=s.tensors[site];s.tensors[site]=c.embedding.unpack_source(ghost)
    physical=s.state_vector()
    assert (s.target_two_j+1)*np.vdot(physical,physical).real<1e-11
    s.tensors[site]=original
    h=np.zeros((8,8));g=np.zeros((8,)*4)
    for i in range(8):g[i,i,i,i]=12.
    for r in range(4):h[2*r,2*r+1]=h[2*r+1,2*r]=-1.
    for r in range(3):
        for leg in range(2):
            i,j=2*r+leg,2*r+2+leg;h[i,j]=h[j,i]=-1.
            i,j=2*r+leg,2*r+3-leg;h[i,j]=h[j,i]=.25
    mpo=ElectronicProblem(h,g,(4,3)).su2_mpo();c.set_hamiltonian(mpo.native_mpo(s.physical_basis))
    p=SimpleNamespace(local_dimension=c.rank,apply_hamiltonian=c.apply,apply_metric=lambda x:x,metric_scale=1.)
    x=c.embedding.pack_source(original)
    e,z,_=_davidson_in_metric_coordinates(p,LETTADMROptions(),c.coordinates(x))
    assert e==pytest.approx(-3.13657157753983,abs=2e-10)
    s.tensors[site]=c.embedding.unpack_source(c.expand(z))
    assert _energy(s,mpo,stable=True)==pytest.approx(e,abs=2e-10)
