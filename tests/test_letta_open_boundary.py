import itertools

import numpy as np
import pytest

from pyqed._letta_one_site_opt.open_boundary import (
    OpenState, OpenContractions, OpenOneSiteOptions, open_one_site, whiten_open_site,
)
from pyqed._letta_one_site_opt.periodic import hubbard_ring_terms, bose_hubbard_ring_terms
from test_letta_periodic import dense_operator, fock_hubbard


def state_for(variant, dimension=4):
    s = OpenState.random(5,dimension,seed=47,complex_values=True)
    if variant != 'mps':
        s = s.with_nn_ties(wrap=variant=='wrap')
        rng = np.random.default_rng(63)
        s.tensors = [a*(1+.2*rng.normal(size=a.shape)) for a in s.tensors]
    return s


@pytest.mark.parametrize('variant',['mps','letta','wrap'])
def test_open_contractions_and_all_local_operators(variant):
    state = state_for(variant)
    configs = np.array(list(itertools.product(range(4),repeat=5)))
    h = dense_operator(hubbard_ring_terms(5),5)
    np.testing.assert_allclose(h.toarray(),fock_hubbard(5),atol=0)
    cache = OpenContractions(state,hubbard_ring_terms(5))
    v = state.amplitudes(configs)
    np.testing.assert_allclose(cache.norm(),np.vdot(v,v),atol=1e-11)
    np.testing.assert_allclose(cache.energy(),np.vdot(v,h@v)/np.vdot(v,v),atol=1e-11)
    assert np.linalg.norm(v[np.array([0,1,1,2])[configs].sum(axis=1)!=5]) == 0
    for site in (0,2,4):
        original = state.tensors[site].copy()
        indices = np.flatnonzero(state.mask(site).ravel())
        jac = []
        for j in indices:
            state.tensors[site] = np.zeros_like(original)
            state.tensors[site].flat[j] = 1
            jac.append(state.amplitudes(configs))
        state.tensors[site] = original
        jac = np.array(jac).T
        heff,neff = cache.local_matrices(site)
        np.testing.assert_allclose(neff[np.ix_(indices,indices)],jac.conj().T@jac,atol=2e-11)
        np.testing.assert_allclose(heff[np.ix_(indices,indices)],jac.conj().T@(h@jac),atol=2e-10)


@pytest.mark.parametrize('variant',['mps','letta','wrap'])
def test_open_gauge_preserves_complex_state_and_charge_masks(variant):
    state = state_for(variant,6)
    configs = np.array(list(itertools.product(range(4),repeat=5)))
    before = state.amplitudes(configs)
    for site in (0,4,2,1,3):
        cache = OpenContractions(state,())
        whiten_open_site(state,site,cache.local_term(0,site))
        np.testing.assert_allclose(state.amplitudes(configs),before,rtol=1e-10,atol=1e-12)
        assert all(np.all(a[~state.mask(i)]==0) for i,a in enumerate(state.tensors))
        if variant=='mps':
            n = OpenContractions(state,()).local_term(0,site)
            ix = np.flatnonzero(state.mask(site).ravel())
            n = n[np.ix_(ix,ix)]
            diag = np.diag(n).real
            n = n/np.sqrt(diag[:,None]*diag[None,:])
            np.testing.assert_allclose(n,np.eye(len(ix)),atol=1e-10)


@pytest.mark.parametrize('numbers',[(0,1,1,2),(0,1,2)])
def test_open_paired_starts_and_variational_sweeps(numbers):
    initial = OpenState.random(5,3,particle_numbers=numbers)
    config = np.array(list(itertools.product(range(len(numbers)),repeat=5)))
    terms = hubbard_ring_terms(5) if len(numbers)==4 else bose_hubbard_ring_terms(5)
    for state in (initial,initial.with_nn_ties(),initial.with_nn_ties(wrap=True)):
        np.testing.assert_allclose(state.amplitudes(config),initial.amplitudes(config),atol=1e-14)
        final,history,_ = open_one_site(state,terms,OpenOneSiteOptions(max_sweeps=4))
        assert np.max(np.diff([r['energy'] for r in history])) <= 1e-10
        assert all(r['gauge_rejections']==0 and r['rejected_steps']==0 for r in history[1:])
        if len(numbers)==4:
            from pyqed._letta_one_site_opt.benchmarks.periodic_hubbard_validation import physical_check
            checked = physical_check(final)['physical_energy']
        else:
            from pyqed._letta_one_site_opt.benchmarks.periodic_bose_validation import sector_hamiltonian
            configs,h = sector_hamiltonian(5)
            v = final.amplitudes(configs)
            checked = np.vdot(v,h@v)/np.vdot(v,v)
        np.testing.assert_allclose(history[-1]['energy'],checked,atol=1e-10)


@pytest.mark.parametrize('variant',['mps','letta','wrap'])
def test_scalar_balance_preserves_very_unequal_tensor_scales(variant):
    from pyqed._letta_one_site_opt.open_boundary import balance_tensor_scales
    state = state_for(variant)
    config = np.array(list(itertools.product(range(4),repeat=5)))
    before = state.amplitudes(config)
    for a,power in zip(state.tensors,[40,-30,20,-50,20]):
        a *= 10.**power
    balance_tensor_scales(state)
    np.testing.assert_allclose(state.amplitudes(config),before,atol=1e-12,rtol=1e-11)
    norms = [np.linalg.norm(a) for a in state.tensors]
    assert max(norms)/min(norms) < 1+1e-12


def test_independent_open_support_hamiltonian_matches_physical_periodic_model():
    from pyqed._letta_one_site_opt.benchmarks.open_boundary_support import fermion_sector
    configs,h = fermion_sector(5)
    indices = configs @ (4**np.arange(4,-1,-1))
    np.testing.assert_allclose(h.toarray(),fock_hubbard(5)[np.ix_(indices,indices)],atol=0)


def test_adding_wrap_tie_to_nontrivial_open_letta_is_exact_embedding():
    state = state_for('letta')
    tensors = [a.copy() for a in state.tensors]
    tensors[-1] = np.repeat(tensors[-1][:,None],state.physical_dim,axis=1)
    wrapped = OpenState(tensors,'letta',state.charges,state.particle_numbers,wrap_tie=True)
    configs = np.array(list(itertools.product(range(4),repeat=5)))
    np.testing.assert_allclose(wrapped.amplitudes(configs),state.amplitudes(configs),atol=1e-14)
