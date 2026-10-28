"""Shared pair actions and growth directions must preserve the physical space."""
import numpy as np
import pytest
from collections import Counter
from pyqed._letta_one_site_opt import ReducedLatticeLETTA, ReducedFrontier, LETTADMROptions
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.reduced_solver import _energy, optimize_reduced_site
from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg
from pyqed._letta_two_site_opt.reduced_solver import _optimize_allocated_reduced_pair, _select_multiplet_ranks
from pyqed.mps.nonabelian.states import build_reduced_product_spatial_mps, build_random_reduced_spatial_mps
from test_letta_qchem import integrals


def test_inactive_tie_entries_survive_one_site_update():
    p=ElectronicProblem(*integrals(4),(2,2))
    state=ReducedLatticeLETTA.from_mps(
        build_reduced_product_spatial_mps(['double','double','empty','empty']),
        symmetry=p.symmetry('su2'),neighborhoods=((0,2),(1,),(2,),(3,)))
    e=ReducedFrontier.from_state(state).site_embedding(state,0)
    before=e.source_layout.pack(state.tensors[0])
    inactive=np.setdiff1d(np.arange(e.source_layout.size),e.supported_source_indices)
    assert len(inactive)>0 and np.linalg.norm(before[inactive])>0
    h=p.su2_mpo();energy=_energy(state,h,stable=True)
    optimize_reduced_site(state,h,0,LETTADMROptions())
    np.testing.assert_array_equal(e.source_layout.pack(state.tensors[0])[inactive],before[inactive])
    assert _energy(state,h,stable=True)==pytest.approx(energy,abs=2e-12)


def test_growth_retains_zero_weight_multiplets_within_total_budget():
    from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
    a,b=SpinChargeSector(0,SU2Irrep(0)),SpinChargeSector(2,SU2Irrep(0))
    dec={a:dict(singular_values=np.array([1.,0.]),available=2),
         b:dict(singular_values=np.array([0.]),available=1)}
    ranks=_select_multiplet_ranks(dec,3,keep_zero_multiplets=True)
    assert ranks=={a:2,b:1}
    assert sum(_select_multiplet_ranks(dec,2,keep_zero_multiplets=True).values())==2
    assert _select_multiplet_ranks(dec,3)=={a:1,b:0}


def test_pair_outside_ties_matches_equivalent_mps():
    p=ElectronicProblem(*integrals(5),(3,2))
    state=ReducedLatticeLETTA.random((1,5),symmetry=p.symmetry('su2'),
        neighborhoods=((0,1),(1,),(2,),(3,),(4,)),seed=710,real=False)
    sites=ReducedFrontier.from_state(state).to_mps(state,compact=True)
    mps=ReducedLatticeLETTA.from_mps(sites,symmetry=state.symmetry,
        neighborhoods=tuple((i,) for i in range(5)))
    h=p.su2_mpo();options=LETTATwoSiteOptions(split_method='metric-als-energy',
        eigensolver_tolerance=1e-11,metric_tolerance=1e-12)
    a=_optimize_allocated_reduced_pair(state,h,3,'lr',6,options)
    b=_optimize_allocated_reduced_pair(mps,h,3,'lr',6,options)
    assert a.accepted and b.accepted
    assert a.energy==pytest.approx(b.energy,abs=2e-10)
    assert a.local_energy==pytest.approx(b.local_energy,abs=2e-10)
    assert a.conditional_discarded_weight==pytest.approx(b.conditional_discarded_weight,abs=2e-10)
    assert a.truncation_iterations==b.truncation_iterations==0
    np.testing.assert_allclose(state.state_vector(),mps.state_vector(),atol=2e-9)


@pytest.mark.parametrize('ties,product_seed',[(1,True),(1,False),(2,False)])
def test_h4_qchem_tied_growth_reaches_fci_and_new_spin(ties,product_seed,monkeypatch):
    pyscf=pytest.importorskip('pyscf')
    from pyscf import gto,scf,ao2mo,fci
    from test_letta_pair_support import restricted_bond
    mol=gto.M(atom=[('H',(0,0,1.2*i)) for i in range(4)],basis='sto-3g',verbose=0)
    mf=scf.RHF(mol).run(conv_tol=1e-12);c=mf.mo_coeff
    h1=c.T@mf.get_hcore()@c;eri=ao2mo.restore(1,ao2mo.kernel(mol,c),4)
    p=ElectronicProblem(h1,eri,(2,2),mol.energy_nuc());sym=p.symmetry('su2')
    exact=fci.direct_spin1.kernel(h1,eri,4,(2,2),ecore=p.ecore,tol=1e-12)[0]
    sites=(build_reduced_product_spatial_mps(['double','double','empty','empty']) if product_seed
        else build_random_reduced_spatial_mps(4,target_sector=sym.sector,bond_multiplicity=1,seed=91))
    state=ReducedLatticeLETTA.from_mps(sites,symmetry=sym,
        neighborhoods=((0,2),(1,3) if ties==2 else (1,),(2,),(3,)))
    if not product_seed:state=restricted_bond(state,1,lambda q:q.irrep.two_j!=2)
    assert all(q.irrep.two_j!=2 for q in state.bond_sectors[1])
    def forbidden(*a,**kw):raise AssertionError('production solver expanded a determinant state')
    monkeypatch.setattr(ReducedLatticeLETTA,'state_vector',forbidden)
    result=letta_two_site_dmrg(p.su2_mpo(),state=state,bond_dim=12,
        options=LETTATwoSiteOptions(max_sweeps=8,tolerance=1e-12,
            eigensolver_tolerance=1e-11,metric_tolerance=1e-12,
            reduced_sector_growth=True,split_method='metric-als-energy',dense_solver_threshold=32))
    assert result.energy==pytest.approx(exact,abs=1e-9)
    assert any(q.irrep.two_j==2 for q in result.state.bond_sectors[1])
    assert max(map(len,result.state.bond_sectors))<=12
    assert any(u.compression_diagnostics and u.compression_diagnostics['used_solver']=='als'
               for row in result.history for u in row.updates)
    assert result.state.symmetry_violation()==0.


@pytest.mark.parametrize('complex_state',[False,True])
def test_plain_pair_fresh_residual_avoids_repeating_a_stationary_solve(complex_state,monkeypatch):
    import pyqed._letta_two_site_opt.reduced_solver as pair
    p=ElectronicProblem(*integrals(5),(3,2))
    initial=ReducedLatticeLETTA.random((1,5),symmetry=p.symmetry('su2'),
        neighborhoods=((0,1),(1,),(2,),(3,),(4,)),seed=710,real=not complex_state)
    h=p.su2_mpo();options=LETTATwoSiteOptions(split_method='metric-als-energy',
        eigensolver_tolerance=1e-11,metric_tolerance=1e-12)
    a=initial.copy()
    update=pair._optimize_allocated_reduced_pair(a,h,3,'lr',200,options)
    assert update.accepted and update.energy_refinement_iterations==0
    certificate=update.energy_refinement_diagnostics
    assert certificate['converged']
    assert certificate['stationarity'][0]['relative_residual']<=options.eigensolver_tolerance
    monkeypatch.setattr(pair,'_plain_pair_refinement_certificate',lambda *a,**kw:None)
    b=initial.copy()
    reference=pair._optimize_allocated_reduced_pair(b,h,3,'lr',200,options)
    assert reference.energy_refinement_iterations>=1
    assert update.energy==pytest.approx(reference.energy,abs=2e-10)
    assert _energy(a,h,stable=True)==pytest.approx(_energy(b,h,stable=True),abs=2e-10)
    va,vb=a.state_vector(),b.state_vector()
    assert abs(np.vdot(va,vb))/(np.linalg.norm(va)*np.linalg.norm(vb))==pytest.approx(1,abs=2e-10)


def test_plain_pair_nonstationary_compression_still_refines():
    import pyqed._letta_two_site_opt.reduced_solver as pair
    p=ElectronicProblem(*integrals(4),(2,2))
    state=ReducedLatticeLETTA.random((1,4),symmetry=p.symmetry('su2'),
        neighborhoods=tuple((i,) for i in range(4)),seed=19,real=True)
    update=pair._optimize_allocated_reduced_pair(state,p.su2_mpo(),1,'lr',3,
        LETTATwoSiteOptions(split_method='metric-als-energy',eigensolver_tolerance=1e-11))
    assert update.energy_refinement_iterations>0
    assert update.conditional_discarded_weight>1e-8


@pytest.mark.parametrize('ties',[0,1,2])
@pytest.mark.parametrize('complex_state',[False,True])
def test_packed_canonical_pair_action_matches_direct_contraction(ties,complex_state):
    from pyqed._letta_two_site_opt.reduced_solver import reduced_pair_problem,_orthogonal_pair_action
    from pyqed._letta_two_site_opt.reduced_compression import pair_metric_root
    from pyqed._letta_one_site_opt.reduced_symmetry import _sector_irrep
    p=ElectronicProblem(*integrals(4),(2,2))
    nb=((0,2) if ties else (0,), (1,3) if ties==2 else (1,), (2,), (3,))
    state=ReducedLatticeLETTA.random((1,4),symmetry=p.symmetry('su2'),
        neighborhoods=nb,seed=173,real=not complex_state)
    h=p.su2_mpo()
    problem=reduced_pair_problem(state,h,1,matrix_free=True,norm_only=True,
                                 prepare_canonical_hamiltonian=True)
    root=pair_metric_root(problem,state,1e-12)
    apply=_orthogonal_pair_action(problem,root,state,h)
    rng=np.random.default_rng(12);vector=rng.normal(size=root.size)
    if complex_state:vector=vector+1j*rng.normal(size=root.size)
    weights={k:np.sqrt(_sector_irrep(k[-1]).dim) for k in root.physical_layout.keys}
    blocks=root.physical_layout.unpack(root.embed_physical(vector))
    output=root.environment._pair_action_reference(1,{k:v/weights[k] for k,v in blocks.items()})
    expected=root.project_physical(root.physical_layout.pack({k:output[k]/weights[k] for k in blocks}))
    np.testing.assert_allclose(apply(vector),expected,atol=3e-11,rtol=1e-11)
    assert _orthogonal_pair_action(problem,root,state,h) is apply


@pytest.mark.parametrize('ties',[0,1,2])
@pytest.mark.parametrize('complex_state',[False,True])
def test_supported_pair_diagonal_matches_full_coordinate_action(ties,complex_state):
    from pyqed._letta_two_site_opt.reduced_solver import reduced_pair_problem,_orthogonal_pair_action
    from pyqed._letta_two_site_opt.reduced_compression import pair_metric_root
    p=ElectronicProblem(*integrals(4),(2,2))
    nb=((0,2) if ties else (0,), (1,3) if ties==2 else (1,), (2,), (3,))
    state=ReducedLatticeLETTA.random((1,4),symmetry=p.symmetry('su2'),
        neighborhoods=nb,seed=294,real=not complex_state)
    h=p.su2_mpo()
    problem=reduced_pair_problem(state,h,1,matrix_free=True,norm_only=True,
                                 prepare_canonical_hamiltonian=True)
    root=pair_metric_root(problem,state,1e-12)
    action=_orthogonal_pair_action(problem,root,state,h)
    # Explicit reduced coordinate columns are only a small-system test oracle.
    expected=np.array([action(e)[j] for j,e in enumerate(np.eye(root.size))])
    np.testing.assert_allclose(action.diagonal(),expected,rtol=1e-12,atol=2e-11)
    np.testing.assert_allclose(action.diagonal().imag,0.,atol=2e-11)
