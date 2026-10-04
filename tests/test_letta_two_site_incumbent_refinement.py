"""A truncated merged root must not discard a better fixed-rank relaxation."""

import numpy as np

from pyqed._letta_one_site_opt import LETTADMROptions, letta_dmrg
from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
from pyqed._letta_one_site_opt.benchmarks.condensed_runner import make_shared_initial_state
from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg


def test_two_site_avoids_bose_hubbard_split_plateau():
    model = build_model("bose_hubbard", dimension="2d", size=(2, 2))
    initial = make_shared_initial_state(model, bond_dim=2, seed=731).letta
    before = initial.state_vector().copy()
    one = letta_dmrg(
        model.mpo, state=initial,
        options=LETTADMROptions(max_sweeps=100, tolerance=1e-9),
    )
    two = letta_two_site_dmrg(
        model.mpo, state=initial, bond_dim=2,
        options=LETTATwoSiteOptions(max_sweeps=16, tolerance=1e-9),
    )

    # Refining only the split stops after three passes at -11.4241009318,
    # even though the same fixed-rank ansatz reaches about -11.69709.
    assert two.energy <= one.energy + 1e-4
    vector = two.state.state_vector()
    hamiltonian = model.mpo.to_dense()  # Independent small-system reference only.
    physical_energy = np.vdot(vector, hamiltonian @ vector) / np.vdot(vector, vector)
    np.testing.assert_allclose(two.energy, physical_energy, atol=2e-10, rtol=0)
    np.testing.assert_array_equal(initial.state_vector(), before)
    assert two.state.bond_dimensions == (2, 2, 2)
    assert two.polish_sweeps == 0
    assert np.max(np.diff([initial.expectation(model.mpo)] + [s.energy for s in two.history])) < 2e-10

    first = two.history[0].updates[0]
    assert first.energy_refinement_start == "incumbent"
    assert first.incumbent_refinement_energy < first.split_refinement_energy - 0.4
    for sweep in two.history:
        for update in sweep.updates:
            assert update.energy <= min(
                update.old_energy,
                update.incumbent_refinement_energy,
                update.split_refinement_energy,
            ) + 2e-10
