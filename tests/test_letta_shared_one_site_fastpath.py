"""Exact shared one-site contractions and gauge certificates, including ties."""
import numpy as np
import pytest
from pyqed._letta_one_site_opt.reduced_environment import _advance_single_physical
from pyqed._letta_two_site_opt.sector_operator import SectorOperator, prepare_sector_operator


@pytest.mark.parametrize('complex_values', [False, True])
@pytest.mark.parametrize('lr', [False, True])
@pytest.mark.parametrize('dims', [(2,3,4,5,2,7), (5,2,2,3,7,2)])
def test_single_physical_absorption_matches_four_factor_reference(complex_values, lr, dims):
    rng=np.random.default_rng(20261027)
    def draw(shape):
        x=rng.normal(size=shape)
        return x+1j*rng.normal(size=shape) if complex_values else x
    x,y,a,l,b,r=dims
    bra,ket,kernel=draw((a,1,b)),draw((l,1,r)),draw((x,y,1,1))
    incoming=draw((x,a,l) if lr else (y,b,r))
    eq='xal,xyop,aob,lpr->ybr' if lr else 'ybr,xyop,aob,lpr->xal'
    expected=np.einsum(eq,incoming,kernel,bra.conj(),ket,optimize=True)
    np.testing.assert_allclose(_advance_single_physical(incoming,kernel,bra,ket,lr),
                               expected,rtol=2e-13,atol=2e-13)


@pytest.mark.parametrize('complex_values', [False, True])
@pytest.mark.parametrize('budget', [1, 32*1024**2])
@pytest.mark.parametrize('channels', [(7,2), (2,7)])
def test_precontracted_channels_keep_all_physical_copies_and_complex_actions(complex_values,budget,channels):
    rng=np.random.default_rng(463)
    def draw(shape):
        x=rng.normal(size=shape)
        return x+1j*rng.normal(size=shape) if complex_values else x
    shapes={0:(3,2,4),1:(5,3,2),2:(0,1,3)}
    x,y=channels
    left,right=draw((x,5,3)),draw((y,2,4))
    terms=[(1,0,left,draw((x,y,3,2)),right)]
    # Duplicate boundaries, exact zeros and cancellation are legal.
    w=draw((x,y,3,2));w[::2]=0
    terms.extend([(1,0,left,w,right),(1,0,left,-w,right)])
    new=prepare_sector_operator(shapes,terms,.7,max_bytes=budget)
    old=SectorOperator(shapes,terms,.7)
    for _ in range(2):
        blocks={k:draw(v)+1j*draw(v) for k,v in shapes.items()}
        expected=old(blocks)
        actual=new(blocks)
        for k in shapes:
            np.testing.assert_allclose(actual[k],expected[k],rtol=3e-12,atol=3e-12)
    zero=prepare_sector_operator(shapes,[])
    np.testing.assert_array_equal(zero.apply_packed(np.ones(zero.layout.size)),0)


def test_boundary_signatures_detect_inplace_values_and_new_layouts():
    from pyqed._letta_one_site_opt.frontier_contraction import CanonicalBoundaryCache
    from pyqed._letta_one_site_opt.reduced_environment import ReducedTransfer
    from pyqed.mps.nonabelian.tensor import NonabelianTensor
    from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
    q=SpinChargeSector(0,SU2Irrep(0));key=(q,q,q)
    site=NonabelianTensor({key:np.ones((1,1,1))},[[q],[q],[q]],[-1,1,1])
    kernel=np.ones((1,1,1,1));transfer=ReducedTransfer(key,key,key,key,kernel)
    cache=CanonicalBoundaryCache();mpo=object();ts=((transfer,),)
    signature=cache.signatures([site],mpo,ts)
    assert cache.signatures([site],mpo,ts)==signature
    site.data[key]*=2
    changed=cache.signatures([site],mpo,ts)
    assert changed!=signature
    kernel*=3
    assert cache.signatures([site],mpo,ts)!=changed


@pytest.mark.parametrize('direction', ['lr','rl'])
@pytest.mark.parametrize('tied', [False,True])
def test_plain_gauge_certificate_preserves_state_and_rejects_corruption(direction,tied):
    from pyqed._letta_one_site_opt import ReducedLatticeLETTA, LETTADMROptions
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    from pyqed._letta_one_site_opt.reduced_solver import ReducedSweepContext, _energy
    from pyqed._letta_one_site_opt.reduced_gauge import shift_reduced_frontier_gauge, condition_reduced_sweep
    from test_letta_qchem import integrals
    p=ElectronicProblem(*integrals(5),(3,2))
    neighborhoods=((0,2),(1,),(2,),(3,),(4,)) if tied else tuple((i,) for i in range(5))
    state=ReducedLatticeLETTA.random((1,5),symmetry=p.symmetry('su2'),
        neighborhoods=neighborhoods,multiplets_per_sector=2,real=False,seed=73)
    ham=p.su2_mpo();context=ReducedSweepContext(state,ham);before=state.copy()
    energy=_energy(before,ham,stable=True)
    shift_reduced_frontier_gauge(state,4,direction)
    assert context.certify_gauge(before,4)
    np.testing.assert_allclose(state.state_vector(),before.state_vector(),rtol=2e-12,atol=2e-12)
    assert abs(_energy(state,ham,stable=True)-energy)<2e-11
    next(iter(state.tensors[3].values()))[...] *= 1.2
    assert not context.certify_gauge(before,4)
    if tied:
        assert not context.certify_gauge(before,1)
    state=before.copy();context=ReducedSweepContext(state,ham)
    assert condition_reduced_sweep(state,ham,LETTADMROptions(),cut=4,
                                   direction=direction,context=context) is None
    assert abs(context.stable_energy()-_energy(state,ham,stable=True))<2e-11


def test_source_energy_reuse_invalidates_after_changed_tensor():
    from test_letta_reduced_updates import case
    from pyqed._letta_one_site_opt import LETTADMROptions
    from pyqed._letta_one_site_opt.reduced_solver import ReducedSweepContext, optimize_reduced_site, _energy
    state,ham=case();context=ReducedSweepContext(state,ham)
    update=optimize_reduced_site(state,ham,1,LETTADMROptions(),context=context)
    assert update.accepted
    np.testing.assert_allclose(context.stable_energy(),_energy(state,ham,stable=True),atol=2e-12,rtol=0)
    next(iter(state.tensors[0].values()))[...] *= 1.02
    np.testing.assert_allclose(context.stable_energy(),_energy(state,ham,stable=True),atol=2e-12,rtol=0)


def test_sweep_does_not_build_unused_source_matrices():
    from test_letta_reduced_updates import case
    from pyqed._letta_one_site_opt import LETTADMROptions
    from pyqed._letta_one_site_opt.reduced_solver import ReducedSweepContext, reduced_local_problem
    state,ham=case()
    context=ReducedSweepContext(state,ham)
    problem=context.local_problem(1,LETTADMROptions(matrix_free=False))
    reference=reduced_local_problem(state,ham,1,matrix_free=False)
    assert problem.hamiltonian is None and problem.metric is None
    assert reference.hamiltonian is not None and reference.metric is not None
    vector=np.random.default_rng(452).normal(size=problem.local_dimension)
    np.testing.assert_allclose(problem.apply_hamiltonian(vector),reference.hamiltonian@vector,atol=2e-12)
    np.testing.assert_allclose(problem.apply_metric(vector),reference.metric@vector,atol=2e-12)


def test_zero_channel_operator_and_mixed_environment_dtypes():
    from pyqed._letta_one_site_opt.reduced_environment import ReducedTransfer, advance_left
    from pyqed.mps.nonabelian.tensor import NonabelianTensor
    from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
    shapes={0:(2,1,2)}
    op=prepare_sector_operator(shapes,[(0,0,np.ones((0,2,2)),np.ones((0,1,1,1)),np.ones((1,2,2)))])
    np.testing.assert_array_equal(op.apply_packed(np.ones(4)),0)
    q=SpinChargeSector(0,SU2Irrep(0));p=SpinChargeSector(1,SU2Irrep(0))
    k0,k1=(q,q,q),(q,p,q)
    core=NonabelianTensor({k0:np.ones((1,1,1)),k1:np.ones((1,1,1))*(1+1j)},
                         [[q],[q,p],[q]],[-1,1,1])
    boundary=(q,q,q)
    ts=[ReducedTransfer(k0,k0,boundary,boundary,np.ones((1,1,1,1))),
        ReducedTransfer(k0,k1,boundary,boundary,np.ones((1,1,1,1)))]
    result=advance_left({boundary:np.ones((1,1,1))},core,ts)
    np.testing.assert_allclose(result[boundary],2+1j)


def test_compiled_kernel_owns_immutable_bytes_and_pruning_keeps_them():
    from dataclasses import replace
    from pyqed._letta_one_site_opt.reduced_environment import CompiledReducedTransfer
    source=np.arange(6.).reshape(2,3,1,1)
    transfer=CompiledReducedTransfer((),(),(),(),source)
    before=transfer.immutable_digest()
    import pickle
    loaded=pickle.loads(pickle.dumps(transfer))
    assert loaded.immutable_digest()==before
    with pytest.raises(ValueError):loaded.kernel.setflags(write=True)
    source[...] = 0
    np.testing.assert_array_equal(transfer.kernel,np.arange(6.).reshape(2,3,1,1))
    with pytest.raises(ValueError):transfer.kernel.setflags(write=True)
    with pytest.raises(ValueError):transfer.kernel[0,0]=7
    changed=replace(transfer,kernel=2*transfer.kernel)
    assert changed.immutable_digest()!=before
    assert transfer.immutable_digest()==before
    # Shape metadata belongs to the boundary key; a noncontiguous view must
    # never reuse a digest of the old element order.
    object.__setattr__(changed,'kernel',changed.kernel.transpose(1,0,2,3))
    assert changed.immutable_digest() is None
