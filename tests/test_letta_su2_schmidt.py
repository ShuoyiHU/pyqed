"""Independent physical-norm checks of the untied SU(2) Schmidt split."""
import numpy as np
import pytest

from pyqed._letta_one_site_opt import ReducedLatticeLETTA
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg
from pyqed._letta_two_site_opt.reduced_solver import reduced_pair_problem, _schmidt_split_untied
from test_letta_qchem import integrals


def split_state(state, h, cap):
    problem = reduced_pair_problem(state, h, 1, matrix_free=True, dense_solver_threshold=0)
    sites = problem.frontier.to_mps(state)
    split = _schmidt_split_untied(problem, problem.layout.unpack(problem.old_vector),
        sites, state, cap, 'lr', LETTATwoSiteOptions(metric_tolerance=1e-12))
    result = state.copy()
    result.tensors[1], result.tensors[2] = split.left_blocks, split.right_blocks
    return result, split.discarded_weight


@pytest.mark.parametrize('nelec', [(2, 2), (2, 1)])
@pytest.mark.parametrize('cap', [3, 100])
def test_schmidt_loss_matches_dense_state_and_is_gauge_invariant(nelec, cap):
    p = ElectronicProblem(*integrals(4), nelec)
    s = ReducedLatticeLETTA.random((1, 4), symmetry=p.symmetry('su2'),
        neighborhoods=tuple((i,) for i in range(4)), multiplets_per_sector=2,
        real=False, seed=301)
    h = p.su2_mpo()
    before = s.state_vector()
    result, loss = split_state(s, h, cap)
    after = result.state_vector()
    np.testing.assert_allclose(np.linalg.norm(before-after)**2, loss, atol=2e-11)
    np.testing.assert_allclose(np.vdot(after, before-after), 0., atol=2e-11)
    if cap == 100:
        np.testing.assert_allclose(after, before, atol=2e-11)
    rng = np.random.default_rng(305)
    gauged = s.copy()
    for cut in (1, 3):
        transforms = {}
        for (ql, qp, qr), a in gauged.tensors[cut-1].items():
            if qr not in transforms:
                size = a.shape[-1]
                transforms[qr] = 2*np.eye(size)+.2*(rng.normal(size=(size,size))
                                                   +1j*rng.normal(size=(size,size)))
            gauged.tensors[cut-1][ql,qp,qr] = a@transforms[qr]
        for (ql, qp, qr), a in gauged.tensors[cut].items():
            gauged.tensors[cut][ql,qp,qr] = np.einsum('al,lpb->apb',
                                                    np.linalg.inv(transforms[ql]), a)
    np.testing.assert_allclose(gauged.state_vector(), before, atol=1e-12)
    changed, changed_loss = split_state(gauged, h, cap)
    np.testing.assert_allclose(changed_loss, loss, atol=2e-11)
    np.testing.assert_allclose(changed.state_vector(), after, atol=2e-10)


def test_adaptive_untied_solver_uses_native_schmidt_split(monkeypatch):
    p = ElectronicProblem(*integrals(3), (2, 1))
    s = ReducedLatticeLETTA.random((1, 3), symmetry=p.symmetry('su2'),
        neighborhoods=((0,), (1,), (2,)), seed=71)
    def forbidden(*args, **kwargs):
        raise AssertionError('active MPS solve expanded a global state')
    monkeypatch.setattr(ReducedLatticeLETTA, 'state_vector', forbidden)
    result = letta_two_site_dmrg(p.su2_mpo(), state=s, bond_dim=8,
        options=LETTATwoSiteOptions(max_sweeps=6, tolerance=1e-12,
            split_method='conditional-svd', reduced_sector_growth=True,
            dense_solver_threshold=1, gauge_mode='frontier'))
    from pyscf import fci
    # This random Hamiltonian's lowest M=1/2 state is a quartet. Compare with
    # the independently selected doublet, the actual target of this solve.
    oracle = fci.addons.fix_spin_(fci.direct_spin1.FCI(), shift=10., ss=.75)
    exact, ci = oracle.kernel(p.h1, p.eri, 3, p.nelec)
    assert fci.spin_op.spin_square(ci, 3, p.nelec)[0] == pytest.approx(.75, abs=1e-10)
    assert result.energy == pytest.approx(exact, abs=2e-9)
    assert all(u.truncation_iterations == 0 for x in result.history for u in x.updates)
@pytest.mark.parametrize('side',['left','right'])
def test_schmidt_handles_structurally_disconnected_outer_sector(side):
    p=ElectronicProblem(*integrals(4),(2,2))
    s=ReducedLatticeLETTA.random((1,4),symmetry=p.symmetry('su2'),
        neighborhoods=tuple((i,) for i in range(4)),multiplets_per_sector=2,
        real=False,seed=301)
    index,axis=(0,2) if side=='left' else (3,0)
    q=next(iter(s.tensors[index]))[axis]
    s.tensors[index]={k:v for k,v in s.tensors[index].items() if k[axis]!=q}
    before=s.state_vector()
    assert np.linalg.norm(before)>0
    result,loss=split_state(s,p.su2_mpo(),100)
    np.testing.assert_allclose(result.state_vector(),before,atol=2e-11)
    assert loss==pytest.approx(0,abs=2e-11)


@pytest.mark.parametrize('nelec', [(2, 2), (2, 1)])
@pytest.mark.parametrize('cap', [3, 100])
def test_cached_qr_schmidt_matches_norm_contraction(nelec, cap, monkeypatch):
    from pyqed._letta_one_site_opt.reduced_norm import ReducedNormChain
    p = ElectronicProblem(*integrals(4), nelec)
    state = ReducedLatticeLETTA.random((1, 4), symmetry=p.symmetry('su2'),
        neighborhoods=tuple((i,) for i in range(4)), multiplets_per_sector=2,
        real=False, seed=301)
    # Nonunitary exterior gauges exercise complex Gram conjugation and scale.
    before = state.state_vector()
    rng = np.random.default_rng(305)
    for cut in (1, 3):
        transforms = {}
        for (ql, qp, qr), value in state.tensors[cut-1].items():
            if qr not in transforms:
                n = value.shape[-1]
                transforms[qr] = 2*np.eye(n)+.2*(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))
            state.tensors[cut-1][ql,qp,qr] = value@transforms[qr]
        for (ql, qp, qr), value in state.tensors[cut].items():
            state.tensors[cut][ql,qp,qr] = np.einsum('al,lpb->apb', np.linalg.inv(transforms[ql]), value)
    np.testing.assert_allclose(state.state_vector(), before, atol=2e-11)
    h = p.su2_mpo()
    reference, loss = split_state(state, h, cap)
    problem = reduced_pair_problem(state, h, 1, matrix_free=True,
        norm_only=True, prepare_canonical_hamiltonian=True)
    sites = problem.frontier.to_mps(state)
    def forbidden(*args, **kwargs):
        raise AssertionError('cached pair split rebuilt the norm chain')
    monkeypatch.setattr(ReducedNormChain, 'build', forbidden)
    split = _schmidt_split_untied(problem, problem.layout.unpack(problem.old_vector),
        sites, state, cap, 'lr', LETTATwoSiteOptions(metric_tolerance=1e-12))
    result = state.copy()
    result.tensors[1], result.tensors[2] = split.left_blocks, split.right_blocks
    np.testing.assert_allclose(result.state_vector(), reference.state_vector(), atol=3e-11)
    assert split.discarded_weight == pytest.approx(loss, abs=3e-11)
