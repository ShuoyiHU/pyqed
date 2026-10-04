"""Two-site frontier gauges checked against physical states and QR energies."""

import numpy as np
import pytest

from pyqed._letta_one_site_opt import LatticeLETTA, canonicalize_frontier
from pyqed._letta_one_site_opt.canonical import get_canonical
from pyqed._letta_one_site_opt._letta_for_2d import transverse_field_ising_mpo
from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
from pyqed._letta_one_site_opt.benchmarks.condensed_runner import make_shared_initial_state
from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg


def test_two_site_defaults_to_frontier():
    assert LETTATwoSiteOptions().gauge_mode == "frontier"


@pytest.mark.parametrize("direction", ["lr", "rl"])
@pytest.mark.parametrize("matrix_free", [False, True])
@pytest.mark.parametrize("alternate", [False, True])
def test_frontier_two_site_physical_energy_and_supported_environments(
    direction, matrix_free, alternate,
):
    state = LatticeLETTA.random((1, 4), bond_dim=4, seed=81, real=False)
    h = transverse_field_ising_mpo((1, 4), field=.9)
    original = state.state_vector()
    initial_energy = state.expectation(h)
    result = letta_two_site_dmrg(h, state=state, bond_dim=4, options=LETTATwoSiteOptions(
        gauge_mode="frontier", start_direction=direction, alternate=alternate,
        matrix_free=matrix_free, dense_solver_threshold=1,
        max_sweeps=3, tolerance=1e-13,
    ))
    vector = result.state.state_vector()
    dense = h.to_dense()
    physical_energy = np.real(np.vdot(vector, dense @ vector) / np.vdot(vector, vector))
    np.testing.assert_allclose(result.energy, physical_energy, atol=2e-10)
    np.testing.assert_allclose(result.energy, np.linalg.eigvalsh(dense)[0], atol=2e-9)
    assert np.max(np.diff([initial_energy] + [s.energy for s in result.history])) < 2e-10
    np.testing.assert_allclose(state.state_vector(), original, atol=0, rtol=0)
    assert result.state.bond_dimensions == (4, 4, 4)
    # The outgoing side must actually be canonical, not merely accept the flag.
    last_direction = result.history[-1].direction
    for cut in range(1, 4):
        record = get_canonical(result.state, cut, last_direction)
        assert record is not None
        assert record.report.applied
        assert record.report.projector_residual < 1e-9


@pytest.mark.parametrize("model_name", ["bose_hubbard", "fermi_hubbard"])
def test_hubbard_frontier_preserves_state_and_matches_qr_energy(model_name):
    model = build_model(model_name, dimension="1d", size=3)
    state = make_shared_initial_state(model, bond_dim=4, seed=731).letta
    before = state.state_vector()
    energy_before = state.expectation(model.mpo)
    canonicalize_frontier(state, center=1)
    np.testing.assert_allclose(state.state_vector(), before, atol=2e-13)
    np.testing.assert_allclose(state.expectation(model.mpo), energy_before, atol=2e-12)
    results = [letta_two_site_dmrg(model.mpo, state=state, bond_dim=4,
        options=LETTATwoSiteOptions(gauge_mode=mode, max_sweeps=12, tolerance=1e-11))
        for mode in ("qr", "frontier")]
    exact = np.linalg.eigvalsh(model.mpo.to_dense())[0]
    for result in results:
        np.testing.assert_allclose(result.energy, result.state.expectation(model.mpo), atol=2e-10)
        np.testing.assert_allclose(result.energy, exact, atol=2e-8)
    np.testing.assert_allclose(results[0].energy, results[1].energy, atol=2e-8)
