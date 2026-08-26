from pathlib import Path

import numpy as np

from pyqed.mps.dmrg import reconstruct_state_average_root_factors
from pyqed.qchem.dmrg.dmrg import _take_tensor_dmrg_runtime_options


def _dense_state_vector(factors):
    state = np.asarray(factors[0])
    for factor in factors[1:]:
        state = np.tensordot(state, np.asarray(factor), axes=(-1, 0))
    return state.reshape(-1)


def test_reconstruct_state_average_roots_from_sweep_payload():
    common = [np.ones((1, 2, 1)), np.ones((1, 2, 1))]
    root0 = np.zeros((1, 2, 2, 1))
    root1 = np.zeros((1, 2, 2, 1))
    root0[0, 0, 0, 0] = 1.0
    root1[0, 1, 0, 0] = 1.0

    roots = reconstruct_state_average_root_factors(
        common,
        [root0, root1],
        last_i=0,
        max_bond=2,
        symmetry=False,
    )

    np.testing.assert_allclose(_dense_state_vector(roots[0]), root0.reshape(-1))
    np.testing.assert_allclose(_dense_state_vector(roots[1]), root1.reshape(-1))


def test_qchem_dmrg_forwards_sweep_and_checkpoint_controls():
    callback = object()
    kwargs = {
        "sweep_callback": callback,
        "checkpoint_path": Path("checkpoint.pkl"),
        "resume_from": Path("resume.pkl"),
        "checkpoint_interval": 2,
        "recenter_final": False,
    }

    options = _take_tensor_dmrg_runtime_options(kwargs)

    assert options == {
        "sweep_callback": callback,
        "checkpoint_path": Path("checkpoint.pkl"),
        "resume_from": Path("resume.pkl"),
        "checkpoint_interval": 2,
        "recenter_final": False,
    }
    assert kwargs == {}
