"""Ring CBE residual selection, exact padding, true one-site solves and recovery."""
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from pyqed._letta_one_site_opt import ReducedRingLETTA, ReducedPhysicalBasis, LETTADMROptions, letta_dmrg
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.reduced_ring_cbe import select_ring_cbe, expand_ring_cbe, ring_cbe_site
from pyqed._letta_one_site_opt.reduced_ring_solver import ring_energy, optimize_ring_site
from pyqed._letta_two_site_opt.reduced_ring_allocation import grow_ring_bond
from pyqed._letta_two_site_opt.reduced_solver import _active_source_indices
from pyqed._letta_compression import MetricCompressionOptions
from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
from test_letta_ring_sweeps import direct_conditional_ring
from test_letta_ring_pair import reference_pair_frame
from test_letta_qchem import determinant_hamiltonian, integrals


def incomplete():
    state=ReducedRingLETTA.random(2,ReducedPhysicalBasis.spatial_orbital(),
        SpinChargeSector(2,SU2Irrep(0)),anchor_sector=SpinChargeSector(0,SU2Irrep(1)),
        neighborhoods=((0,1),(1,0)),seed=73)
    middle=next(q for q in state.bond_sectors[1] if q.components[0]==2)
    tensors=[{k:a for k,a in state.tensors[0].items() if k[2]==middle},
             {k:a for k,a in state.tensors[1].items() if k[0]==middle}]
    state=ReducedRingLETTA(state.physical_basis,state.symmetry.sector,tensors,
        bond_sectors=(state.bond_sectors[0],(middle,),state.bond_sectors[2]),
        closure=state.closure,neighborhoods=state.neighborhoods)
    state=grow_ring_bond(state,0,{middle:3})
    p=ElectronicProblem(np.array([[0.,-1.],[-1.,0.]]),np.zeros((2,)*4),(1,1))
    return state,p


def controls(**kwargs):
    return LETTADMROptions(cbe_enabled=True,max_sweeps=4,gauge_mode='none',matrix_free=False,
        cbe_expansion_dimension=1,cbe_refinement_max_iterations=12,
        cbe_projection_max_iterations=200,cbe_energy_refinement_max_iterations=8,
        compression=MetricCompressionOptions(als_max_iterations=30,lsmr_max_iterations=120),**kwargs)


@pytest.mark.parametrize('direction',['lr','rl'])
def test_ring_cbe_discovers_missing_multiplet_and_preserves_padding(direction):
    s,p=incomplete()
    before=direct_conditional_ring(s)
    selection=select_ring_cbe(s,p.su2_mpo(),0,controls())
    assert selection.missing_norm > .1
    assert selection.projection_converged
    assert selection.tangent_overlap < 1e-8
    assert sum(selection.multiplicities.values())==1
    assert any(q not in s.bond_sectors[1] for q in selection.multiplicities)
    expanded=expand_ring_cbe(s,selection,direction)
    assert len(expanded.bond_sectors[1])==4
    np.testing.assert_allclose(direct_conditional_ring(expanded),before,atol=2e-12)
    np.testing.assert_array_equal(direct_conditional_ring(s),before)


@pytest.mark.parametrize('solver',['als','variable-projection','joint-ls','grassmann-newton'])
def test_ring_cbe_all_compressors_beat_same_start_one_site(solver):
    s,p=incomplete();h=p.su2_mpo()
    options=replace(controls(),compression=MetricCompressionOptions(solver=solver,
        als_max_iterations=40,lsmr_max_iterations=120,max_iterations=100))
    baseline=s.copy()
    ordinary=optimize_ring_site(baseline,h,0,replace(options,cbe_enabled=False))
    update=ring_cbe_site(s,h,0,'lr',3,options)
    assert update.cbe_recovery_reason is None,update.cbe_recovery_reason
    assert update.cbe_expanded_energy < ordinary.energy-1e-4
    assert update.energy <= ordinary.energy+1e-10
    assert not update.cbe_baseline_selected
    assert update.cbe_materialized_pair_metric is True
    assert update.cbe_compression_diagnostics[0]['requested_solver']==solver
    assert len(s.bond_sectors[1])<=3
    v=direct_conditional_ring(s)
    assert update.energy==pytest.approx(np.vdot(v,determinant_hamiltonian(p)@v).real/np.vdot(v,v).real,abs=2e-10)


def test_public_ring_cbe_solves_stuck_case_without_pair_diagonalization(monkeypatch):
    import pyqed._letta_two_site_opt.reduced_ring_solver as pair
    import pyqed._letta_two_site_opt.reduced_solver as open_pair
    import pyqed._letta_one_site_opt.reduced_contraction as components
    s,p=incomplete();h=p.su2_mpo()
    before=direct_conditional_ring(s)
    def forbidden(*args,**kwargs):
        raise AssertionError('CBE called a pair eigensolver or expanded magnetic tensor')
    monkeypatch.setattr(pair,'_solve_local_problem',forbidden)
    monkeypatch.setattr(open_pair,'_solve_local_problem',forbidden)
    monkeypatch.setattr(components,'expand_reduced_mps_site',forbidden)
    ordinary=letta_dmrg(h,state=s,options=replace(controls(),cbe_enabled=False))
    result=letta_dmrg(h,state=s,options=controls())
    assert ordinary.energy==pytest.approx(0.,abs=1e-12)
    assert result.energy==pytest.approx(-2.,abs=2e-8)
    assert max(result.state.bond_dimensions)<=3
    assert not any(u.cbe_recovery_reason for row in result.history for u in row.updates)
    assert any(u.cbe_expansion_dimension for row in result.history for u in row.updates)
    assert all(u.energy <= u.cbe_baseline_energy+1e-10 for row in result.history for u in row.updates)
    assert any(u.is_target_closure for row in result.history for u in row.updates)
    np.testing.assert_array_equal(direct_conditional_ring(s),before)


@pytest.mark.parametrize('site',[0,1,2])
@pytest.mark.parametrize('direction',['lr','rl'])
def test_ring_cbe_failure_restores_and_uses_same_start_baseline(monkeypatch,site,direction):
    import pyqed._letta_one_site_opt.reduced_ring_cbe as cbe
    s,p=incomplete();h=p.su2_mpo();options=controls()
    baseline=s.copy()
    ordinary=optimize_ring_site(baseline,h,site,replace(options,cbe_enabled=False))
    def failed(candidate,*args,**kwargs):
        next(iter(candidate.site_blocks(site).values())).flat[0]=np.nan
        candidate.bond_sectors=()
        raise MemoryError('injected ring selector failure')
    monkeypatch.setattr(cbe,'select_ring_cbe',failed)
    update=ring_cbe_site(s,h,site,direction,3,options)
    assert update.cbe_fallback and update.cbe_baseline_selected
    assert not update.local_converged
    assert 'MemoryError' in update.cbe_recovery_reason
    assert update.energy==pytest.approx(ordinary.energy,abs=2e-12)
    assert s.bond_sectors==baseline.bond_sectors
    np.testing.assert_allclose(direct_conditional_ring(s),direct_conditional_ring(baseline),atol=2e-12)


@pytest.mark.parametrize('edge',[0,1,2])
def test_cyclic_missing_residual_matches_independent_physical_projection(edge):
    s=ReducedRingLETTA.random(2,ReducedPhysicalBasis.spatial_orbital(),
        SpinChargeSector(2,SU2Irrep(0)),anchor_sector=SpinChargeSector(0,SU2Irrep(1)),
        neighborhoods=((0,1),(1,0)),seed=16)
    model=ElectronicProblem(*integrals(2),(1,1))
    selection=select_ring_cbe(s,model.su2_mpo(),edge,controls())
    p=selection.problem;scaffold=selection.scaffold
    frame=reference_pair_frame(p)
    le,re=p.left_embedding,p.right_embedding
    a=le.pack_source(scaffold.site_blocks(edge))
    b=re.pack_source(scaffold.site_blocks(p.right_site))
    ranks=Counter(s.bond_sectors[p.right_site])
    li,ri=_active_source_indices(le,ranks,'left'),_active_source_indices(re,ranks,'right')
    columns=[]
    for k in li:
        da=np.zeros_like(a);da[k]=1
        columns.append(frame@p.merge(da,b))
    for k in ri:
        db=np.zeros_like(b);db[k]=1
        columns.append(frame@p.merge(a,db))
    tangent=np.column_stack(columns)
    v=frame@p.old_vector;h=determinant_hamiltonian(model)
    energy=np.vdot(v,h@v).real/np.vdot(v,v).real
    gradient=h@v-energy*v
    pair_gradient=frame@np.linalg.lstsq(frame,gradient,rcond=1e-12)[0]
    missing=pair_gradient-tangent@np.linalg.lstsq(tangent,pair_gradient,rcond=1e-12)[0]
    assert selection.missing_norm==pytest.approx(np.linalg.norm(missing),abs=3e-9)
    assert selection.tangent_overlap < 3e-8
    for direction in ('lr','rl'):
        padded=expand_ring_cbe(s,selection,direction)
        np.testing.assert_allclose(direct_conditional_ring(padded),direct_conditional_ring(s),atol=3e-12)


def test_nonzero_ring_residual_uses_a_nonseparable_cyclic_metric():
    from pyqed.mps.symmetry import Sector
    from pyqed._letta_one_site_opt import LatticeMPO
    q=Sector(('charge','su2'),(0,SU2Irrep(0)))
    s=ReducedRingLETTA.random(4,ReducedPhysicalBasis(('neutral',),(q,),(2,)),q,
        multiplets_per_sector=2,seed=42)
    x=np.array([[0.,1.],[1.,0.]])
    z=np.diag([1.,-1.])
    ops=(x,z,x,np.eye(2))
    h=LatticeMPO(tuple(a[None,None] for a in ops))
    dense=np.kron(np.kron(np.kron(*ops[:2]),ops[2]),ops[3])
    selected=select_ring_cbe(s,h,0,controls())
    p=selected.problem
    frame=reference_pair_frame(p)
    reshuffle=p.metric.reshape(4,4,4,4).transpose(0,2,1,3).reshape(16,16)
    assert np.linalg.matrix_rank(reshuffle,tol=1e-10)>1
    le,re=p.left_embedding,p.right_embedding
    a=le.pack_source(selected.scaffold.site_blocks(0))
    b=re.pack_source(selected.scaffold.site_blocks(1))
    li,ri=_active_source_indices(le,{q:2},'left'),_active_source_indices(re,{q:2},'right')
    columns=[]
    for k in li:
        da=np.zeros_like(a);da[k]=1
        columns.append(frame@p.merge(da,b))
    for k in ri:
        db=np.zeros_like(b);db[k]=1
        columns.append(frame@p.merge(a,db))
    tangent=np.column_stack(columns)
    v=frame@p.old_vector
    energy=np.vdot(v,dense@v).real/np.vdot(v,v).real
    gradient=dense@v-energy*v
    missing=gradient-tangent@np.linalg.lstsq(tangent,gradient,rcond=1e-12)[0]
    assert selected.missing_norm > 1e-3
    assert selected.missing_norm==pytest.approx(np.linalg.norm(missing),abs=3e-9)
    assert selected.captured_weight > 1e-4
    assert selected.tangent_overlap < 2e-8
    for direction in ('lr','rl'):
        candidate=expand_ring_cbe(s,selected,direction)
        np.testing.assert_allclose(direct_conditional_ring(candidate),direct_conditional_ring(s),atol=2e-12)


def test_u1_bose_ring_cbe_preserves_charge_and_reaches_exact_energy():
    from pyqed.mps.symmetry import Sector
    from pyqed._letta_one_site_opt import LatticeMPO
    basis=ReducedPhysicalBasis(('0','1','2'),tuple(
        Sector(('number','su2'),(n,SU2Irrep(0))) for n in range(3)),(1,1,1))
    target=Sector(('number','su2'),(2,SU2Irrep(0)))
    s=ReducedRingLETTA.random(2,basis,target,multiplets_per_sector=2,
        neighborhoods=((0,1),(1,0)),seed=41)
    middle=basis.sectors[2]
    tensors=[{k:a for k,a in s.tensors[0].items() if k[2]==middle},
             {k:a for k,a in s.tensors[1].items() if k[0]==middle}]
    s=ReducedRingLETTA(s.physical_basis,s.symmetry.sector,tensors,
        bond_sectors=(s.bond_sectors[0],(middle,)*2,s.bond_sectors[2]),
        closure=s.closure,neighborhoods=s.neighborhoods)
    s=grow_ring_bond(s,0,{middle:3})
    a=np.diag(np.sqrt([1.,2.]),1);number=np.diag([0.,1.,2.]);identity=np.eye(3)
    onsite=1.5*number@(number-identity)
    h=LatticeMPO((np.stack([-a.T,-a,onsite,identity])[None],
                  np.stack([a,a.T,identity,onsite])[:,None]))
    dense=-np.kron(a.T,a)-np.kron(a,a.T)+np.kron(onsite,identity)+np.kron(identity,onsite)
    ids=[i for i,c in enumerate(np.ndindex(3,3)) if sum(c)==2]
    exact=np.linalg.eigvalsh(dense[np.ix_(ids,ids)])[0]
    result=letta_dmrg(h,state=s,options=controls())
    assert result.energy==pytest.approx(exact,abs=2e-8)
    assert not any(u.cbe_recovery_reason for row in result.history for u in row.updates)
    assert any(u.cbe_expansion_dimension for row in result.history for u in row.updates)
    v=direct_conditional_ring(result.state)
    np.testing.assert_allclose((np.kron(number,identity)+np.kron(identity,number))@v,2*v,atol=2e-12)
    assert np.vdot(v,dense@v).real/np.vdot(v,v).real==pytest.approx(result.energy,abs=2e-9)
    assert max(result.state.bond_dimensions)<=3


def test_ring_cbe_failed_ordinary_step_preserves_all_allocations(monkeypatch):
    import pyqed._letta_one_site_opt.reduced_ring_cbe as cbe
    s,p=incomplete();original=direct_conditional_ring(s);bonds=s.bond_sectors
    def failed(candidate,*args,**kwargs):
        candidate.bond_sectors=()
        raise np.linalg.LinAlgError('ordinary failure')
    monkeypatch.setattr(cbe,'optimize_ring_site',failed)
    update=ring_cbe_site(s,p.su2_mpo(),2,'rl',3,controls())
    assert not update.accepted and not update.local_converged
    assert update.cbe_recovery_rejected
    assert 'ordinary failure' in update.cbe_recovery_reason
    assert s.bond_sectors==bonds
    np.testing.assert_array_equal(direct_conditional_ring(s),original)


def test_complex_three_orbital_qc_ring_with_backward_ties():
    p=ElectronicProblem(*integrals(3),(2,1),.17)
    s=ReducedRingLETTA.random(3,ReducedPhysicalBasis.spatial_orbital(),
        SpinChargeSector(3,SU2Irrep(1)),anchor_sector=SpinChargeSector(0,SU2Irrep(1)),
        neighborhoods=((0,2),(1,),(2,0)),seed=61)
    before=ring_energy(s,p.su2_mpo())
    result=letta_dmrg(p.su2_mpo(),state=s,options=replace(controls(),max_sweeps=1,
        gauge_mode='frontier',cbe_energy_refinement_max_iterations=3))
    vector=direct_conditional_ring(result.state)
    dense=np.kron(determinant_hamiltonian(p),np.eye(2))
    energy=np.vdot(vector,dense@vector).real/np.vdot(vector,vector).real
    assert result.energy==pytest.approx(energy,abs=2e-9)
    assert result.energy <= before+1e-10
    assert result.state.neighborhoods==s.neighborhoods
    assert not any(u.cbe_recovery_reason for row in result.history for u in row.updates)
    from test_letta_qchem_symmetry import total_operators
    number,_,spin=total_operators(3)
    np.testing.assert_allclose(np.kron(number,np.eye(2))@vector,3*vector,atol=2e-10)
    np.testing.assert_allclose(np.kron(spin,np.eye(2))@vector,.75*vector,atol=2e-10)


def test_failed_ring_cbe_trim_preserves_ordinary_baseline(monkeypatch):
    import pyqed._letta_one_site_opt.reduced_ring_cbe as cbe
    s,p=incomplete();h=p.su2_mpo();options=controls()
    baseline=s.copy()
    ordinary=optimize_ring_site(baseline,h,0,replace(options,cbe_enabled=False))
    def failed(candidate,*args,**kwargs):
        next(iter(candidate.closure.data.values())).fill(np.nan)
        candidate.bond_sectors=()
        raise FloatingPointError('injected ring trim failure')
    monkeypatch.setattr(cbe,'fit_ring_pair_target',failed)
    update=ring_cbe_site(s,h,0,'lr',3,options)
    assert update.cbe_expansion_dimension>0
    assert update.cbe_fallback and update.cbe_baseline_selected
    assert 'trim failure' in update.cbe_recovery_reason
    assert update.energy==pytest.approx(ordinary.energy,abs=2e-12)
    np.testing.assert_allclose(direct_conditional_ring(s),direct_conditional_ring(baseline),atol=2e-12)
    assert s.bond_sectors==baseline.bond_sectors


def test_cyclic_root_workspace_limit_uses_real_ordinary_recovery():
    s,p=incomplete();h=p.su2_mpo()
    options=replace(controls(),compression=MetricCompressionOptions(max_workspace_mb=1e-12))
    baseline=s.copy()
    ordinary=optimize_ring_site(baseline,h,0,replace(options,cbe_enabled=False))
    update=ring_cbe_site(s,h,0,'lr',3,options)
    assert update.cbe_fallback and update.cbe_baseline_selected
    assert 'MemoryError' in update.cbe_recovery_reason
    assert update.energy==pytest.approx(ordinary.energy,abs=2e-12)
    np.testing.assert_allclose(direct_conditional_ring(s),direct_conditional_ring(baseline),atol=2e-12)
