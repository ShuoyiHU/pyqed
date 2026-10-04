"""Coupled CBE relaxation must preserve factorization and reach the Bose basin."""
import numpy as np
from pyqed._letta_one_site_opt import LETTADMROptions, letta_dmrg
from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
from pyqed._letta_one_site_opt.benchmarks.condensed_runner import make_shared_initial_state
from pyqed._letta_two_site_opt import LETTAPairLayout


def test_cbe_coupled_relaxation_closes_bose_gap_without_pair_merge(monkeypatch):
    model = build_model('bose_hubbard', '2d', (2, 2))
    initial = make_shared_initial_state(model, bond_dim=2, seed=731).letta
    before = initial.state_vector().copy()
    def forbidden(*args, **kwargs):
        raise AssertionError('strict CBE must not merge a pair')
    monkeypatch.setattr(LETTAPairLayout, 'merge', forbidden)
    result = letta_dmrg(model.mpo, state=initial, options=LETTADMROptions(
        cbe_enabled=True, cbe_selector='shrewd', metric_tolerance=1e-10,
        tolerance=1e-12, max_sweeps=40))
    assert result.converged
    np.testing.assert_allclose(result.energy, -11.69709376305754, atol=2e-10, rtol=0)
    vector = result.state.state_vector()
    physical = np.vdot(vector, model.mpo.to_dense() @ vector) / np.vdot(vector, vector)
    np.testing.assert_allclose(result.energy, physical, atol=2e-10, rtol=0)
    np.testing.assert_array_equal(initial.state_vector(), before)
    assert result.state.bond_dimensions == initial.bond_dimensions
    assert max(np.diff([initial.expectation(model.mpo)] + [s.energy for s in result.history])) < 2e-10


import pytest
from dataclasses import replace
from pyqed._letta_one_site_opt import (
    AbelianSymmetry, LatticeLETTA, IdentityEnvironmentCache, LETTAEnvironmentCache,
)
from pyqed._letta_one_site_opt._letta_for_2d import transverse_field_ising_mpo
from pyqed._letta_one_site_opt.cbe_coupled import (
    _cross_overlap, _conditioned_direction, refine_coupled_factors,
)


@pytest.mark.parametrize('shape', [(1, 4), (2, 2), (2, 2, 2)])
@pytest.mark.parametrize('real', [False, True])
def test_factor_cross_overlap_matches_physical_reference(shape, real, monkeypatch):
    state = LatticeLETTA.random(shape, bond_dim=2, seed=127, real=real)
    cache = IdentityEnvironmentCache(state)
    le, re = cache.build_left_environments(), cache.build_right_environments()
    def forbidden(*a, **kw):
        raise AssertionError('pair merge is forbidden')
    monkeypatch.setattr(LETTAPairLayout, 'merge', forbidden)
    for site in range(state.nsites - 1):
        layout = LETTAPairLayout.from_state(state, site)
        actual = _cross_overlap(cache, le[site], re[site+2], layout,
                                *state.tensors[site:site+2])
        expected = state.local_frame(site).conj().T @ state.local_frame(site+1)
        np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=3e-13)


def _context(complex_values=False, restricted=False):
    symmetry = AbelianSymmetry((0, 1), sector=0, moduli=2) if restricted else None
    state = LatticeLETTA.random((2, 2), bond_dim=2, seed=713, symmetry=symmetry)
    if complex_values:
        rng = np.random.default_rng(722)
        state.tensors = [a * np.exp(1j*rng.normal(size=a.shape)) for a in state.tensors]
    mpo = transverse_field_ising_mpo((2, 2), coupling=0.7, field=1.2, basis='x')
    layout = LETTAPairLayout.from_state(state, 0)
    hc, nc = LETTAEnvironmentCache(state, mpo), IdentityEnvironmentCache(state)
    hr, nr = hc.build_right_environments(), nc.build_right_environments()
    arguments = (state, layout, hc, nc, hc.scalar_boundary(), hr[2], nc.scalar_boundary(), nr[2])
    return arguments, mpo


@pytest.mark.parametrize('complex_values', [False, True])
@pytest.mark.parametrize('restricted', [False, True])
def test_coupled_factors_preserve_energy_symmetry_and_input(complex_values, restricted, monkeypatch):
    args, mpo = _context(complex_values, restricted)
    state, layout, hc = args[:3]
    initial = [a.copy() for a in state.tensors]
    v = state.state_vector()
    h = mpo.to_dense()  # Independent validation only.
    before = np.vdot(v, h @ v).real / np.vdot(v, v).real
    applications = []
    original = hc.prepare_effective_action
    def prepare(*a, **kw):
        action = original(*a, **kw)
        def counted(v):
            applications.append(1)
            return action(v)
        return counted
    def forbidden(*a, **kw):
        raise AssertionError('no physical frame or merged pair in CBE')
    monkeypatch.setattr(hc, 'prepare_effective_action', prepare)
    monkeypatch.setattr(LETTAPairLayout, 'merge', forbidden)
    monkeypatch.setattr(state, 'local_frame', forbidden)
    result = refine_coupled_factors(*args, options=LETTADMROptions(
        cbe_coupled_max_iterations=5, cbe_coupled_activation_ratio=0.01), norm_limit=100)
    assert 0 < result.accepted_steps <= result.iterations <= 5
    assert result.hamiltonian_applications == len(applications) == 2*result.iterations
    for actual, expected in zip(state.tensors, initial):
        np.testing.assert_array_equal(actual, expected)
    state.tensors[:2] = [result.left, result.right]
    v = state.state_vector()
    physical = np.vdot(v, h @ v).real / np.vdot(v, v).real
    assert physical < before - 1e-5
    np.testing.assert_allclose(result.energy, physical, atol=3e-12, rtol=0)
    np.testing.assert_allclose(result.norm, np.linalg.norm(v), atol=1e-12, rtol=0)
    if restricted:
        assert state.symmetry_violation() == 0


def test_coupled_failure_restores_original_tensor_objects(monkeypatch):
    args, _ = _context()
    state, layout, hc = args[:3]
    saved = list(state.tensors)
    def fail(*a, **kw):
        raise RuntimeError('action failure')
    monkeypatch.setattr(hc, 'prepare_effective_action', fail)
    with pytest.raises(RuntimeError, match='action failure'):
        refine_coupled_factors(*args, options=LETTADMROptions(), norm_limit=100)
    assert all(a is b for a,b in zip(saved, state.tensors))


def test_impossible_factor_budget_rejects_all_steps():
    args, mpo = _context()
    result = refine_coupled_factors(*args, options=LETTADMROptions(
        cbe_coupled_activation_ratio=0.01), norm_limit=1e-30)
    assert result.accepted_steps == 0
    assert result.iterations == 1
    np.testing.assert_allclose(result.energy, args[0].expectation(mpo), atol=1e-12, rtol=0)


@pytest.mark.parametrize('dtype', [np.float32,np.float64,np.complex64,np.complex128])
@pytest.mark.parametrize('size,boundary', [(0,0),(3,0),(3,2)])
def test_empty_or_zero_support_returns_zero_direction(dtype,size,boundary):
    delta, residual, independent = _conditioned_direction(
        np.zeros((size,size),dtype=dtype), np.ones(size,dtype=dtype), boundary,1e-12)
    np.testing.assert_array_equal(delta,np.zeros(size))
    assert residual == independent == 0


def test_rank_deficient_bose_tangent_has_finite_descent():
    from pathlib import Path
    path = Path(__file__).parent / 'data/letta/coupled_bose22_rank_deficient.npz'
    with np.load(path) as data:
        metric, gradient = data['metric'], data['gradient']
        delta, residual, independent = _conditioned_direction(
            metric, gradient, int(data['boundary']), float(data['cutoff']))
    assert np.all(np.isfinite(delta))
    assert np.isfinite(residual) and residual > 0
    assert np.isfinite(independent) and independent > 0
    assert np.vdot(gradient, delta).real < 0
    np.testing.assert_allclose(np.vdot(delta, metric @ delta).real,
                               residual, atol=1e-9, rtol=2e-6)


@pytest.mark.parametrize('field,value', [
    ('cbe_coupled_max_iterations',-1), ('cbe_coupled_max_iterations',1.5),
    ('cbe_coupled_max_parameters',-1), ('cbe_coupled_max_parameters',3.5),
    ('cbe_coupled_metric_tolerance',0), ('cbe_coupled_metric_tolerance',1),
    ('cbe_coupled_metric_tolerance',np.nan), ('cbe_coupled_activation_ratio',0),
    ('cbe_coupled_energy_threshold',np.inf), ('cbe_coupled_energy_threshold',-1),
])
def test_invalid_controls_fail_early(field,value):
    model = build_model('heisenberg','2d',(2,2))
    with pytest.raises(ValueError,match=field):
        letta_dmrg(model.mpo,lattice_shape=(2,2),bond_dim=2,
                   options=LETTADMROptions(**{field:value}))


def test_parameter_cap_skips_dense_correction_and_matches_disabled(monkeypatch):
    from pyqed._letta_one_site_opt import cbe_coupled
    model = build_model('ising','2d',(2,2))
    state = make_shared_initial_state(model,bond_dim=1,seed=732).letta
    options = LETTADMROptions(cbe_enabled=True,cbe_selector='shrewd',max_sweeps=2,
                              cbe_coupled_energy_threshold=1.0)
    def forbidden(*a,**kw):
        raise AssertionError('parameter cap must prevent dense correction')
    monkeypatch.setattr(cbe_coupled,'refine_coupled_factors',forbidden)
    capped = letta_dmrg(model.mpo,state=state,options=replace(options,cbe_coupled_max_parameters=0))
    disabled = letta_dmrg(model.mpo,state=state,options=replace(options,cbe_coupled_max_iterations=0))
    np.testing.assert_array_equal([s.energy for s in capped.history], [s.energy for s in disabled.history])
    assert all(u.cbe_coupled_iterations == 0 for s in capped.history for u in s.updates)
