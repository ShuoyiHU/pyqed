import itertools
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import eigsh
from pyqed._letta_one_site_opt.periodic import (
    PeriodicState, RingContractions, hubbard_ring_terms, balance_site, balance_bond,
    PeriodicOneSiteOptions, periodic_one_site,
)


def configurations(length):
    return np.array(list(itertools.product(range(4), repeat=length)))


def dense_operator(terms, length):
    result = sparse.csr_matrix((4**length, 4**length), dtype=float)
    for term in terms:
        factor = sparse.csr_matrix([[term.coefficient]])
        for site in range(length):
            factor = sparse.kron(factor, term.operators.get(site, np.eye(4)), format='csr')
        result += factor
    return result


def fock_hubbard(length):
    """Independent occupation-bit hopping with fermionic signs."""
    matrix = np.zeros((4**length, 4**length))
    config = configurations(length)
    lookup = {tuple(s): i for i, s in enumerate(config)}
    for column, digits in enumerate(config):
        bits = sum(((s & 1) << (2*i)) + (((s >> 1) & 1) << (2*i+1))
                   for i, s in enumerate(digits))
        matrix[column, column] = 4 * sum(s == 3 for s in digits)
        for i in range(length):
            j = (i+1) % length
            for sigma in (0, 1):
                for target, source in ((2*i+sigma, 2*j+sigma), (2*j+sigma, 2*i+sigma)):
                    if not bits & (1 << source) or bits & (1 << target):
                        continue
                    intermediate = bits ^ (1 << source)
                    sign = (-1)**((bits & ((1 << source)-1)).bit_count()
                                 + (intermediate & ((1 << target)-1)).bit_count())
                    final = intermediate | (1 << target)
                    row = lookup[tuple((final >> (2*k)) & 3 for k in range(length))]
                    matrix[row, column] -= sign
    return matrix


def test_periodic_hubbard_has_correct_boundary_fermion_sign():
    np.testing.assert_allclose(dense_operator(hubbard_ring_terms(5), 5).toarray(), fock_hubbard(5), atol=0)


@pytest.mark.parametrize('kind', ['mps', 'letta'])
def test_closed_ring_norm_energy_and_active_metric_match_physical_reference(kind):
    s = PeriodicState.random(5, 2, complex_values=True)
    if kind == 'letta':
        s = s.with_nn_ties()
        rng = np.random.default_rng(13)
        s.tensors = [a * (1 + .1 * rng.normal(size=a.shape)) for a in s.tensors]
    terms = hubbard_ring_terms(5)
    cache = RingContractions(s, terms)
    configs = configurations(5)
    v = s.amplitudes(configs)
    h = dense_operator(terms, 5)
    np.testing.assert_allclose(cache.norm(), np.vdot(v, v).real, atol=1e-12)
    np.testing.assert_allclose(cache.energy(), np.vdot(v, h@v)/np.vdot(v, v), atol=1e-11)
    assert np.linalg.norm(v[np.sum(np.array([0, 1, 1, 2])[configs], axis=1) != 5]) == 0
    site = 4  # Tests the last-first physical tie and virtual closure.
    original = s.tensors[site].copy()
    selected = np.flatnonzero(s.mask.ravel())
    jacobian = []
    for index in selected:
        s.tensors[site] = np.zeros_like(original)
        s.tensors[site].flat[index] = 1
        jacobian.append(s.amplitudes(configs))
    s.tensors[site] = original
    jacobian = np.array(jacobian).T
    heff, neff = cache.local_matrices(site)
    np.testing.assert_allclose(neff[np.ix_(selected, selected)], jacobian.conj().T@jacobian, atol=1e-11)
    np.testing.assert_allclose(heff[np.ix_(selected, selected)], jacobian.conj().T@(h@jacobian), atol=1e-10)


@pytest.mark.parametrize('kind', ['mps', 'letta'])
def test_charge_preserving_conditional_gauges_leave_wavefunction_identical(kind):
    state = PeriodicState.random(5, 4, seed=12, complex_values=True)
    if kind == 'letta':
        state = state.with_nn_ties()
    config = configurations(5)
    original = state.amplitudes(config)
    for site in (0, 4, 2):
        cache = RingContractions(state, ())
        balance_site(state, site, cache.local_term(0, site))
        assert all(np.all(a[~state.mask] == 0) for a in state.tensors)
        np.testing.assert_allclose(state.amplitudes(config), original, atol=2e-12, rtol=1e-11)


def test_same_state_start_and_variational_one_site_sweeps():
    initial = PeriodicState.random(5, 2)
    tied = initial.with_nn_ties()
    config = configurations(5)
    np.testing.assert_allclose(initial.amplitudes(config), tied.amplitudes(config), atol=1e-14)
    terms = hubbard_ring_terms(5)
    h = dense_operator(terms, 5)
    half = np.sum(np.array([0, 1, 1, 2])[config], axis=1) == 5
    exact = eigsh(h[half][:, half], k=1, which='SA', return_eigenvectors=False)[0]
    for state in (initial, tied):
        result, history, _ = periodic_one_site(state, terms, PeriodicOneSiteOptions(max_sweeps=3))
        energies = [r['energy'] for r in history]
        assert np.max(np.diff(energies)) <= 1e-10
        assert energies[-1] >= exact-1e-9
        v = result.amplitudes(config)
        np.testing.assert_allclose(energies[-1], np.vdot(v, h@v)/np.vdot(v,v), atol=1e-10)


@pytest.mark.parametrize('kind', ['mps', 'letta'])
def test_ten_site_periodic_energy_agrees_with_independent_fixed_number_action(kind):
    from pyqed._letta_one_site_opt.benchmarks.periodic_hubbard_validation import physical_check
    state = PeriodicState.random(10, 2, seed=43)
    if kind == 'letta':
        state = state.with_nn_ties()
        rng = np.random.default_rng(18)
        state.tensors = [a*(1+.2*rng.normal(size=a.shape)) for a in state.tensors]
    physical = physical_check(state)
    cache = RingContractions(state, hubbard_ring_terms(10))
    np.testing.assert_allclose(cache.energy(), physical['physical_energy'], atol=1e-10)
    np.testing.assert_allclose(cache.norm(), physical['physical_norm'], atol=1e-10)


def test_sector_reference_matches_independent_full_half_filled_hamiltonian():
    from pyqed._letta_one_site_opt.benchmarks.periodic_hubbard import sector_reference
    config = configurations(5)
    half = np.sum(np.array([0,1,1,2])[config],axis=1)==5
    h = sparse.csr_matrix(fock_hubbard(5))[half][:,half]
    expected = eigsh(h,k=1,which='SA',return_eigenvectors=False)[0]
    np.testing.assert_allclose(sector_reference(5)['energy'],expected,atol=1e-11)


@pytest.mark.parametrize('kind', ['mps', 'letta'])
def test_exact_bond_rebalancing_removes_large_gauge_scales_without_truncation(kind):
    state = PeriodicState.random(5,4,complex_values=True)
    if kind=='letta': state=state.with_nn_ties()
    configs=configurations(5)
    original=state.amplitudes(configs)
    scales=np.array([1e8,1e-8,3.,.2])
    state.tensors[4] *= scales
    state.tensors[0] /= scales[:,None]
    before=sum(np.linalg.norm(state.tensors[i])**2 for i in (4,0))
    for _ in range(4):
        balance_bond(state,4)
    after=sum(np.linalg.norm(state.tensors[i])**2 for i in (4,0))
    assert after < before*1e-10
    assert state.tensors[4].shape[-1]==4
    np.testing.assert_allclose(state.amplitudes(configs),original,atol=2e-12,rtol=1e-11)


def test_rank_deficient_pair_gauge_preserves_neighbor_one_site_search_space():
    state=PeriodicState.random(5,4).with_nn_ties()
    selected=np.flatnonzero(state.mask.ravel())
    def ranks():
        cache=RingContractions(state,())
        return [np.linalg.matrix_rank(cache.local_term(0,i)[np.ix_(selected,selected)],tol=1e-10)
                for i in (0,1)]
    before=ranks()
    balance_bond(state,0)
    assert ranks()==before


def test_complex_padding_preserves_initial_state_and_nested_sectors():
    state=PeriodicState.random(5,4,complex_values=True)
    padded=state.padded_start(6,noise=0.)
    config=configurations(5)
    np.testing.assert_allclose(padded.amplitudes(config),state.amplitudes(config),atol=1e-12)
    assert np.array_equal(padded.charges[:4],state.charges)


@pytest.mark.parametrize('kind', ['mps', 'letta'])
def test_bose_periodic_metric_gauges_and_variational_steps(kind):
    from pyqed._letta_one_site_opt.periodic import bose_hubbard_ring_terms
    from pyqed._letta_one_site_opt.benchmarks.periodic_bose_validation import sector_hamiltonian
    state = PeriodicState.random(5, 3, complex_values=True, particle_numbers=(0,1,2))
    if kind == 'letta':
        state = state.with_nn_ties()
        rng = np.random.default_rng(72)
        state.tensors = [a*(1+.1*rng.normal(size=a.shape)) for a in state.tensors]
    configs,h = sector_hamiltonian(5)
    terms = bose_hubbard_ring_terms(5)
    v = state.amplitudes(configs)
    cache = RingContractions(state,terms)
    np.testing.assert_allclose(cache.norm(),np.vdot(v,v).real,atol=1e-12)
    np.testing.assert_allclose(cache.energy(),np.vdot(v,h@v)/np.vdot(v,v),atol=1e-11)
    site=4
    original=state.tensors[site].copy()
    selected=np.flatnonzero(state.mask.ravel())
    jac=[]
    for index in selected:
        state.tensors[site]=np.zeros_like(original)
        state.tensors[site].flat[index]=1
        jac.append(state.amplitudes(configs))
    state.tensors[site]=original
    jac=np.array(jac).T
    heff,neff=cache.local_matrices(site)
    np.testing.assert_allclose(neff[np.ix_(selected,selected)],jac.conj().T@jac,atol=1e-11)
    np.testing.assert_allclose(heff[np.ix_(selected,selected)],jac.conj().T@(h@jac),atol=1e-10)
    balance_site(state,site,neff)
    balance_bond(state,site)
    np.testing.assert_allclose(state.amplitudes(configs),v,atol=1e-11)
    padded=state.padded_start(4,noise=0)
    np.testing.assert_allclose(padded.amplitudes(configs),v/np.linalg.norm(v),atol=1e-11)
    result,history,_=periodic_one_site(state,terms,PeriodicOneSiteOptions(max_sweeps=3))
    assert np.max(np.diff([r['energy'] for r in history])) <= 1e-10
    v=result.amplitudes(configs)
    np.testing.assert_allclose(history[-1]['energy'],np.vdot(v,h@v)/np.vdot(v,v),atol=1e-10)
    assert history[-1]['energy'] >= eigsh(h,k=1,which='SA',return_eigenvectors=False)[0]-1e-9


def test_bose_ten_site_reference_and_matched_start():
    from pyqed._letta_one_site_opt.periodic import bose_hubbard_ring_terms
    from pyqed._letta_one_site_opt.benchmarks.periodic_bose_validation import sector_hamiltonian
    configs,h=sector_hamiltonian(10)
    state=PeriodicState.random(10,2,particle_numbers=(0,1,2))
    v=state.amplitudes(configs)
    tied=state.with_nn_ties()
    np.testing.assert_allclose(tied.amplitudes(configs),v,atol=1e-12)
    for s in (state,tied):
        np.testing.assert_allclose(RingContractions(s,bose_hubbard_ring_terms(10)).energy(),
                                   np.vdot(v,h@v)/np.vdot(v,v),atol=1e-11)


def test_bose_d4_conditioned_gauge_reaches_valid_ground_state():
    from pyqed._letta_one_site_opt.periodic import bose_hubbard_ring_terms
    from pyqed._letta_one_site_opt.benchmarks.periodic_bose_validation import sector_hamiltonian
    state=PeriodicState.random(5,4,particle_numbers=(0,1,2)).with_nn_ties()
    state,history,converged=periodic_one_site(
        state,bose_hubbard_ring_terms(5),
        PeriodicOneSiteOptions(max_sweeps=30,gauge_floor=1e-3))
    configs,h=sector_hamiltonian(5)
    v=state.amplitudes(configs)
    physical=float(np.vdot(v,h@v).real/np.vdot(v,v).real)
    exact=eigsh(h,k=1,which='SA',return_eigenvectors=False)[0]
    assert converged
    assert history[-1]['max_tensor_norm']<100
    np.testing.assert_allclose(history[-1]['energy'],physical,atol=1e-10,rtol=0)
    np.testing.assert_allclose(physical,exact,atol=1e-10,rtol=0)


@pytest.mark.parametrize('direction', [1,-1])
@pytest.mark.parametrize('kind', ['mps','letta'])
def test_paper_directional_gauge_preserves_complex_state_and_charge(direction,kind):
    from pyqed._letta_one_site_opt.periodic import directional_gauge
    state=PeriodicState.random(5,4,complex_values=True)
    if kind=='letta':state=state.with_nn_ties()
    configs=configurations(5)
    before=state.amplitudes(configs)
    directional_gauge(state,4,direction)
    np.testing.assert_allclose(state.amplitudes(configs),before,atol=2e-11)
    assert all(np.all(a[~state.mask]==0) for a in state.tensors)
    if kind=='mps':
        a=state.tensors[4]
        gram=(np.einsum('sab,sac->bc',a.conj(),a) if direction==1
              else np.einsum('sab,scb->ac',a,a.conj()))
        np.testing.assert_allclose(gram,np.eye(4),atol=1e-11)


def test_paper_gauge_does_not_delete_rank_deficient_one_site_directions():
    from pyqed._letta_one_site_opt.periodic import directional_gauge
    state=PeriodicState.random(5,4).with_nn_ties()
    selected=np.flatnonzero(state.mask.ravel())
    def ranks():
        cache=RingContractions(state,())
        return [np.linalg.matrix_rank(cache.local_term(0,i)[np.ix_(selected,selected)],tol=1e-10)
                for i in (0,1)]
    before=ranks()
    directional_gauge(state,0,1)
    assert ranks()==before


@pytest.mark.parametrize('kind',['mps','letta'])
def test_bose_paper_gauge_sweeps_against_independent_energy(kind):
    from pyqed._letta_one_site_opt.periodic import bose_hubbard_ring_terms
    from pyqed._letta_one_site_opt.benchmarks.periodic_bose_validation import sector_hamiltonian
    state=PeriodicState.random(5,4,particle_numbers=(0,1,2))
    if kind=='letta':state=state.with_nn_ties()
    state,history,_=periodic_one_site(state,bose_hubbard_ring_terms(5),
                                    PeriodicOneSiteOptions(max_sweeps=10,gauge_method='paper'))
    configs,h=sector_hamiltonian(5)
    v=state.amplitudes(configs)
    np.testing.assert_allclose(history[-1]['energy'],np.vdot(v,h@v)/np.vdot(v,v),atol=1e-10,rtol=0)
    assert max(np.diff([r['energy'] for r in history]))<1e-9
