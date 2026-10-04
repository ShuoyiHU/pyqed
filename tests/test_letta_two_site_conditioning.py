"""A split must not bypass the incumbent's factor-growth budget."""

from pathlib import Path

import numpy as np
import pytest

from pyqed._letta_one_site_opt import LatticeLETTA
from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
from pyqed._letta_two_site_opt import (
    IdentityPairEnvironmentCache,
    LETTAPairEnvironmentCache,
    LETTAPairLayout,
    LETTATwoSiteOptions,
)
from pyqed._letta_two_site_opt.solver import _optimize_pair


@pytest.mark.parametrize("pair_gauge_scale", [1.0, 2.0**20, 2.0**-20])
def test_ill_conditioned_split_does_not_raise_physical_energy(pair_gauge_scale):
    # Sweep 1960, reverse pair (2, 3), D2, seed 731. Saving the small local
    # input avoids replaying 15 minutes of sweeps in this regression.
    path = Path(__file__).parent / "data/letta_two_site_ising_2x3_late_pair.npz"
    with np.load(path, allow_pickle=False) as data:
        tensors = [data[f"tensor_{i}"].copy() for i in range(6)]
        boundaries = [data[name].copy() for name in (
            "hamiltonian_left", "hamiltonian_right", "metric_left", "metric_right"
        )]
    tensors[2] *= pair_gauge_scale
    tensors[3] /= pair_gauge_scale
    state = LatticeLETTA((2, 3), physical_dim=2, tensors=tensors)
    # The constructor balances tensors. Restore the recorded local gauge.
    state.tensors = tensors
    mpo = build_model("ising", "2d", (2, 3)).mpo
    hamiltonian = mpo.to_dense()  # Independent validation only.

    def physical_energy():
        vector = state.state_vector()
        return float(np.vdot(vector, hamiltonian @ vector).real
                     / np.vdot(vector, vector).real)

    before = physical_energy()
    np.testing.assert_allclose(before, -8.434692318082876, atol=2e-12, rtol=0)
    update = _optimize_pair(
        state, LETTAPairLayout.from_state(state, 2),
        LETTAPairEnvironmentCache(state, mpo), IdentityPairEnvironmentCache(state),
        *boundaries, 2, "rl", LETTATwoSiteOptions(metric_tolerance=1e-10),
    )
    after = physical_energy()
    assert after <= before + 2e-10
    np.testing.assert_allclose(update.energy, after, atol=2e-9, rtol=0)
    assert update.max_factor_norm < 100
    assert state.bond_dimensions == (2, 2, 2, 2, 2)
    assert update.energy_refinement_start == "incumbent"
    # BLAS backends can produce a better-conditioned ALS split; the physical
    # acceptance requirement is the same for both numerical trajectories.
    assert update.max_factor_norm < update.energy_refinement_factor_norm_limit
