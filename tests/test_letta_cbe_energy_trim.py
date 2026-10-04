"""Independent dense-energy check of benchmark energy ALS in both directions."""
import inspect
from unittest.mock import patch
import numpy as np
from pyqed._letta_one_site_opt import cbe
from pyqed._letta_one_site_opt.benchmarks.cbe_energy_trim import energy_polish
from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
from pyqed._letta_one_site_opt.benchmarks.condensed_runner import run_benchmark


def test_energy_polish_matches_dense_energy_and_restores_live_state():
    model = build_model('fermi_hubbard', '1d', 4)
    h = model.mpo.to_dense(max_sites=4)
    original = cbe._strict_shrewd_cbe_bond_update
    signature = inspect.signature(original)
    checked = set()
    def update(*args, **kwargs):
        context = signature.bind(*args, **kwargs).arguments
        direction = context['direction']
        if direction not in checked:
            state = context['state']
            i = context['layout'].left_site
            before = [t.copy() for t in state.tensors]
            vector = state.state_vector()
            old_energy = np.vdot(vector, h @ vector).real / np.vdot(vector, vector).real
            left, right, norm, details = energy_polish(context, before[i], before[i+1], 2)
            for a,b in zip(state.tensors, before):
                np.testing.assert_array_equal(a,b)
            try:
                state.tensors[i], state.tensors[i+1] = left, right
                vector = state.state_vector()
                exact_energy = np.vdot(vector, h @ vector).real / np.vdot(vector, vector).real
                np.testing.assert_allclose(exact_energy, details['final_energy'], atol=1.e-10, rtol=0)
                np.testing.assert_allclose(norm, np.linalg.norm(vector), atol=1.e-10, rtol=0)
                assert exact_energy <= old_energy + 1.e-10
                assert left.shape == before[i].shape and right.shape == before[i+1].shape
            finally:
                state.tensors[:] = before
            checked.add(direction)
        return original(*args, **kwargs)
    with patch.object(cbe, '_strict_shrewd_cbe_bond_update', update):
        run_benchmark('fermi_hubbard', dimension='1d', size=4, bond_dim=2,
                      max_sweeps=2, solvers=['letta_cbe_strict'], exact_max_dimension=0,
                      raise_on_failure=True)
    assert checked == {'lr', 'rl'}
