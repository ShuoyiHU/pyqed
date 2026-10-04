"""Regressions for energy stationarity after fixed-rank pair truncation."""

import numpy as np
import pytest

from pyqed._letta_one_site_opt import LETTADMROptions, letta_dmrg
from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
from pyqed._letta_one_site_opt.benchmarks.condensed_runner import make_shared_initial_state
from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg


@pytest.mark.parametrize(
    "model_name,length",
    [("heisenberg", 6), ("ising", 6), ("blume_capel", 4), ("spin1_heisenberg", 4)],
)
def test_default_two_site_recovers_one_site_energy_after_als(model_name, length):
    model = build_model(model_name, dimension="1d", size=length)
    initial = make_shared_initial_state(model, bond_dim=2, seed=731).letta
    original = initial.state_vector().copy()
    one = letta_dmrg(
        model.mpo, state=initial,
        options=LETTADMROptions(max_sweeps=60, tolerance=1e-11),
    )
    two = letta_two_site_dmrg(
        model.mpo, state=initial, bond_dim=2,
        options=LETTATwoSiteOptions(max_sweeps=60, tolerance=1e-11),
    )

    # These capped-rank cases are not exact-diagonalization ground states.
    # The target is the variational minimum reached by one-site optimization.
    np.testing.assert_allclose(two.energy, one.energy, atol=2e-9, rtol=0)
    np.testing.assert_allclose(two.energy, two.state.expectation(model.mpo), atol=2e-10)
    np.testing.assert_array_equal(initial.state_vector(), original)
    assert two.polish_sweeps == 0
    assert two.state.bond_dimensions == (2,) * (length - 1)
    assert np.max(np.diff([initial.expectation(model.mpo)] + [s.energy for s in two.history])) < 2e-10
    updates = [u for sweep in two.history for u in sweep.updates]
    assert all(u.truncation_iterations >= 1 for u in updates)
    assert all(u.energy_refinement_iterations >= 1 for u in updates)
    assert all(u.energy_refinement_energy <= u.energy_refinement_initial_energy + 2e-10 for u in updates)
    assert all(u.energy <= u.old_energy + 2e-10 for u in updates)


def test_als_only_can_stall_above_the_variational_minimum():
    model = build_model("spin1_heisenberg", dimension="1d", size=4)
    initial = make_shared_initial_state(model, bond_dim=2, seed=731).letta
    als = letta_two_site_dmrg(
        model.mpo, state=initial, bond_dim=2,
        options=LETTATwoSiteOptions(max_sweeps=60, tolerance=1e-11, split_method="metric-als"),
    )
    refined = letta_two_site_dmrg(
        model.mpo, state=als.state, bond_dim=2,
        options=LETTATwoSiteOptions(max_sweeps=60, tolerance=1e-11),
    )
    assert als.converged
    assert refined.energy < als.energy - 9e-3
    assert all(u.energy_refinement_iterations == 0 for s in als.history for u in s.updates)
