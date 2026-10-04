"""Production custom ties, including backward and environment-only bridges."""
import numpy as np
import pytest

from pyqed._letta_one_site_opt import LatticeLETTA, LETTADMROptions, letta_dmrg
from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg

# Every site owns its first argument; the rest include nonlocal/backward ties.
TIES = ((0, 2, 3), (1, 3), (2, 0), (3, 1))


def _amplitudes(state):
    values = []
    for physical in np.ndindex(*((state.physical_dim,) * state.nsites)):
        value = np.ones((1, 1), dtype=complex)
        for site, tensor in enumerate(state.tensors):
            section = (slice(None),) + tuple(physical[p] for p in state.site_neighborhood(site)) + (slice(None),)
            value = value @ tensor[section]
        values.append(value[0, 0])
    return np.asarray(values)


def test_custom_ties_survive_copy_and_bond_expansion():
    state = LatticeLETTA.random((2, 2), bond_dim=2, neighborhoods=TIES, real=False, seed=1501)
    reference = _amplitudes(state)
    for other in (state, state.copy(), state.without_symmetry(), state.expand_bond_dimension(3)):
        assert other.neighborhoods == TIES
        np.testing.assert_allclose(other.state_vector(), reference, atol=1e-12, rtol=0)
        np.testing.assert_allclose(other.norm(), np.linalg.norm(reference), atol=1e-12)


@pytest.mark.parametrize("ties", [((0,),), ((0, 0), (1,), (2,), (3,)),
                                    ((1,), (1,), (2,), (3,)),
                                    ((0, 4), (1,), (2,), (3,)),
                                    ((0, 1.5), (1,), (2,), (3,))])
def test_custom_ties_are_validated(ties):
    with pytest.raises(ValueError, match="neighborhood"):
        LatticeLETTA.random((2, 2), neighborhoods=ties)


@pytest.mark.parametrize("method", ["one", "cbe", "two"])
@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_all_solvers_preserve_general_ties_and_physical_energy(method, direction):
    state = LatticeLETTA.random((2, 2), bond_dim=1, neighborhoods=TIES, real=False, seed=1502)
    before = _amplitudes(state)
    model = build_model("ising", dimension="2d", size=(2, 2))
    common = dict(max_sweeps=2, tolerance=1e-13, start_direction=direction)
    if method == "two":
        result = letta_two_site_dmrg(model.mpo, state=state, bond_dim=1,
                                     options=LETTATwoSiteOptions(**common))
    else:
        result = letta_dmrg(model.mpo, state=state, options=LETTADMROptions(
            **common, cbe_enabled=method == "cbe", cbe_selector="shrewd"))
    assert result.state.neighborhoods == TIES
    vector = _amplitudes(result.state)
    physical = np.real(np.vdot(vector, model.mpo.to_dense() @ vector) / np.vdot(vector, vector))
    np.testing.assert_allclose(result.energy, physical, atol=2e-9, rtol=0)
    assert result.energy <= state.expectation(model.mpo) + 1e-9
    np.testing.assert_array_equal(_amplitudes(state), before)
