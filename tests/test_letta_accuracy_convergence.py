"""Accuracy checks from a difficult start and nested variational spaces."""

import numpy as np

from pyqed._letta_one_site_opt import LETTADMROptions, letta_dmrg
from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg
from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
from pyqed._letta_one_site_opt.benchmarks.condensed_runner import make_shared_initial_state


def test_two_site_difficult_start_retains_scaled_coupled_directions():
    model = build_model('bose_hubbard', '2d', (2, 2))
    initial = make_shared_initial_state(model, bond_dim=2, seed=1735).letta
    result = letta_two_site_dmrg(model.mpo, state=initial, bond_dim=2,
        options=LETTATwoSiteOptions(max_sweeps=100, tolerance=1e-12,
            eigensolver_tolerance=1e-11, energy_refinement_max_iterations=32))
    vector = result.state.state_vector()
    hamiltonian = model.mpo.to_dense()  # Independent validation only.
    physical = float(np.vdot(vector, hamiltonian @ vector).real / np.vdot(vector, vector).real)
    assert result.converged
    np.testing.assert_allclose(physical, -11.69709376305754, atol=5e-10, rtol=0)
    np.testing.assert_allclose(result.energy, physical, atol=2e-10, rtol=0)


def test_cbe_relaxes_difficult_bose_start_and_two_site_preserves_nested_state():
    model = build_model('bose_hubbard', '2d', (2, 2))
    initial = make_shared_initial_state(model, bond_dim=2, seed=1735).letta
    result = letta_dmrg(model.mpo, state=initial, options=LETTADMROptions(
        max_sweeps=100, tolerance=1e-12, metric_tolerance=1e-10,
        eigensolver_tolerance=1e-11, cbe_enabled=True, cbe_selector='shrewd'))
    assert result.converged
    reference = -11.69709376305754  # Best known D2 energy, not an exact-ground certificate.
    np.testing.assert_allclose(result.energy, reference, atol=5e-10, rtol=0)
    original = result.state.state_vector()
    hamiltonian = model.mpo.to_dense()  # Independent validation only.
    np.testing.assert_allclose(result.energy, np.vdot(original, hamiltonian @ original)
                               / np.vdot(original, original), atol=2e-10, rtol=0)
    for bond in (2, 3):
        expanded = result.state.copy()
        if bond > 2:
            expanded.expand_bond_dimension(bond, noise=0.)
        np.testing.assert_allclose(expanded.state_vector(), original, atol=2e-12, rtol=0)
        optimized = letta_two_site_dmrg(model.mpo, state=expanded, bond_dim=bond,
            options=LETTATwoSiteOptions(max_sweeps=4, tolerance=1e-12,
                                       eigensolver_tolerance=1e-11))
        vector = optimized.state.state_vector()
        physical = float((np.vdot(vector, hamiltonian @ vector) / np.vdot(vector, vector)).real)
        assert physical <= result.energy + 2e-10
        np.testing.assert_allclose(optimized.energy, physical, atol=2e-10, rtol=0)
