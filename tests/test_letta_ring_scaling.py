"""Long cyclic products must survive cancelling, exactly binary core gauges."""
import numpy as np
import pytest

from pyqed.mps.su2 import SU2Irrep
from pyqed.mps.symmetry import Sector
from pyqed.mps.nonabelian.tensor import NonabelianTensor
from pyqed._letta_one_site_opt import ReducedPhysicalBasis, ReducedRingLETTA
from pyqed._letta_one_site_opt.reduced_ring_target import ReducedRingTarget
from pyqed._letta_one_site_opt.reduced_ring_contraction import CyclicReducedNorm, CyclicReducedOperator
from pyqed._letta_one_site_opt.reduced_mpo_compile import SpinTensorMPO
from pyqed._letta_two_site_opt.reduced_ring_pair import CyclicPairProblem


def long_ring(n):
    zero = Sector(('charge', 'su2'), (0, SU2Irrep(0)))
    virtual = Sector(('charge', 'su2'), (0, SU2Irrep(1)))
    basis = ReducedPhysicalBasis(('neutral',), (zero,), (2,))
    rng = np.random.default_rng(127)
    physical = []
    for i in range(n):
        matrices = []
        for p in range(2):
            z = rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2))
            matrices.append(np.linalg.qr(z)[0]/np.sqrt(2))
        a = np.stack(matrices, axis=1)
        physical.append(NonabelianTensor(data={(virtual, zero, virtual): a},
            qns=[(virtual,)*2, (zero,)*2, (virtual,)*2], dirs=[-1, 1, 1],
            metadata={'physical_basis': 'fully_reduced_su2'}))
    closure = NonabelianTensor(data={(virtual, zero, virtual): np.eye(2)[:, None, :]},
        qns=[(virtual,)*2, (zero,), (virtual,)*2], dirs=[-1, 1, 1],
        metadata={'physical_basis': 'fully_reduced_su2'})
    state = ReducedRingLETTA.from_target_ring(ReducedRingTarget(tuple(physical), closure, zero), basis)
    operator = np.array([[.9, .1j], [-.1j, 1.1]])
    mpo = SpinTensorMPO.compile(tuple(operator[None, None] for _ in range(n)), basis)
    # Independent multiplicity transfer, with the spin identity traced in
    # amplitude and conjugate amplitude: (2S+1)^2 = 4.
    prod_n = prod_h = np.eye(4, dtype=complex)
    for site in physical:
        a = next(iter(site.data.values()))
        tn = sum(np.kron(a[:, p, :].conj(), a[:, p, :]) for p in range(2))
        th = sum(operator[p, q]*np.kron(a[:, p, :].conj(), a[:, q, :])
                 for p in range(2) for q in range(2))
        prod_n = prod_n@tn
        prod_h = prod_h@th
    return state, mpo, 4*np.trace(prod_n), 4*np.trace(prod_h)


def rescale(state):
    # Products around the ring overflow/underflow partway through, even though
    # every single- or two-core complement and the total norm are representable.
    n = state.nsites
    powers = [80]*(n//2)+[-80]*(n//2)+[0]
    candidate = state.copy()
    for i, exponent in enumerate(powers):
        candidate.set_site_blocks(i, {k: a*2.**exponent for k, a in state.site_blocks(i).items()})
    return candidate, powers


@pytest.mark.parametrize('n', [32, 64])
def test_long_cyclic_norm_h_and_local_actions_preserve_binary_gauges(n):
    state, mpo, expected_n, expected_h = long_ring(n)
    base_sites = state.to_target_ring().sites
    extended = state.to_target_ring().extend_hamiltonian(mpo)
    baseline_n = CyclicReducedNorm(base_sites)
    baseline_h = CyclicReducedOperator(base_sites, extended)
    np.testing.assert_allclose(baseline_n.overlap(), expected_n, rtol=3e-13, atol=1e-14)
    np.testing.assert_allclose(baseline_h.overlap(), expected_h, rtol=3e-13, atol=1e-14)
    scaled, powers = rescale(state)
    sites = scaled.to_target_ring().sites
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        norm = CyclicReducedNorm(sites)
        h = CyclicReducedOperator(sites, extended)
        np.testing.assert_allclose(norm.overlap(), expected_n, rtol=4e-13, atol=1e-14)
        np.testing.assert_allclose(h.overlap(), expected_h, rtol=4e-13, atol=1e-14)
        for i in (0, n//2, n):
            for chain, baseline in ((norm, baseline_n), (h, baseline_h)):
                actual = chain.local_action(i)
                reference = baseline.local_action(i)
                for key in actual:
                    np.testing.assert_allclose(actual[key]*2.**powers[i], reference[key],
                                               rtol=8e-13, atol=1e-13)


@pytest.mark.parametrize('edge', [0, 15, 31, 32])
def test_long_cyclic_pair_complements_keep_absolute_metric_scale(edge):
    state, mpo, _, _ = long_ring(32)
    scaled, powers = rescale(state)
    reference = CyclicPairProblem(state, mpo, edge)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        actual = CyclicPairProblem(scaled, mpo, edge)
        x = np.random.default_rng(72).normal(size=reference.local_dimension)
        exponent = 2*(powers[edge]+powers[(edge+1)%33])
        for action in ('apply_metric', 'apply_hamiltonian'):
            expected = getattr(reference, action)(x)
            observed = getattr(actual, action)(x)
            np.testing.assert_allclose(observed*2.**exponent, expected, rtol=8e-13, atol=1e-13)


@pytest.mark.parametrize('power', [-80, 80])
def test_unrepresentable_absolute_complement_fails_explicitly(power):
    state, _, _, _ = long_ring(32)
    for i in range(state.nsites):
        state.set_site_blocks(i, {key: a*2.**power for key, a in state.site_blocks(i).items()})
    with pytest.raises(FloatingPointError, match='floating-point range'):
        CyclicReducedNorm(state.to_target_ring().sites).overlap()
