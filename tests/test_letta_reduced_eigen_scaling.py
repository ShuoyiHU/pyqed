"""Local generalized roots must not lose directions under diagonal gauges."""
from types import SimpleNamespace

import numpy as np
import pytest

from pyqed._letta_one_site_opt import LETTADMROptions
from pyqed._letta_one_site_opt.reduced_solver import ReducedLocalProblem, _solve_local_problem


@pytest.mark.parametrize('matrix_free', [False, True])
@pytest.mark.parametrize('scales', [(1., 1., 1.), (1., 1e-12, 1e3), (1e-6, 1e6, 1e-2)])
def test_local_root_preserves_independent_small_norm_direction(scales, matrix_free):
    # Columns 0 and 2 represent one physical direction; column 1 is an
    # independent direction. Rescaling coefficients cannot change the spectrum.
    frame = np.array([[1., 0., 1.], [0., 1., 0.]])*scales
    physical = np.array([[.25, .2], [.2, -1.25]])
    h, n = frame.T@physical@frame, frame.T@frame
    problem = ReducedLocalProblem(0, None,
        SimpleNamespace(source_size=3, target_size=3),
        None if matrix_free else h, None if matrix_free else n,
        hamiltonian_action=lambda x: h@x, metric_action=lambda x: n@x)
    initial = np.array([1/scales[0], 0., 0.])
    energy, vector, rank, _ = _solve_local_problem(problem, LETTADMROptions(), initial_vector=initial)
    expected, directions = np.linalg.eigh(physical)
    actual = frame@vector
    actual /= np.linalg.norm(actual)
    assert energy == pytest.approx(expected[0], abs=2e-12)
    assert rank == 2
    assert abs(np.vdot(actual, directions[:, 0])) == pytest.approx(1., abs=2e-12)
    assert np.vdot(vector, n@vector).real == pytest.approx(1., abs=2e-12)


@pytest.mark.parametrize('pair', [False, True])
def test_native_equilibrated_support_projector_preserves_physical_metric(pair):
    from pyqed._letta_one_site_opt import ReducedLatticeLETTA
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    from pyqed._letta_one_site_opt.reduced_solver import reduced_local_problem
    from pyqed._letta_two_site_opt.reduced_solver import reduced_pair_problem
    p = ElectronicProblem(np.eye(3), np.zeros((3,)*4), (2, 1))
    state = ReducedLatticeLETTA.random((1, 3), symmetry=p.symmetry('su2'),
        multiplets_per_sector=2, seed=37, real=False)
    before = state.state_vector()
    for key, a in state.tensors[0].items():
        state.tensors[0][key] = a*np.geomspace(1e-7, 1e7, a.shape[-1])
    for key, a in state.tensors[1].items():
        scale = np.geomspace(1e-7, 1e7, a.shape[0])
        state.tensors[1][key] = a/scale.reshape((-1,)+(1,)*(a.ndim-1))
    np.testing.assert_allclose(state.state_vector(), before, atol=2e-13)
    problem = (reduced_pair_problem if pair else reduced_local_problem)(state, p.su2_mpo(), 1,
        matrix_free=True, dense_solver_threshold=0)
    projector = problem.equilibrated_projector_factory(1e-12)
    assert projector is not None
    n = np.column_stack([problem.apply_metric(e) for e in np.eye(problem.local_dimension)])
    diagonal = np.diag(n).real
    inverse = np.zeros(len(n))
    positive = diagonal > 0
    inverse[positive] = 1/np.sqrt(diagonal[positive])
    correlation = inverse[:, None]*n*inverse[None, :]
    rng = np.random.default_rng(8)
    x = rng.normal(size=len(n))+1j*rng.normal(size=len(n))
    projected = projector(x)
    np.testing.assert_allclose(correlation@projected, correlation@x, atol=3e-10)
    np.testing.assert_allclose(projector(projected), projected, atol=3e-12)
    assert np.linalg.norm(projected) <= np.linalg.norm(x)+1e-12


@pytest.mark.parametrize('matrix_free', [False, True])
@pytest.mark.parametrize('case', ['diagonal', 'complex-disconnected', 'singular-scaled'])
def test_stationary_excited_start_does_not_hide_lower_root(case, matrix_free):
    # The incumbent is an exact excited eigenstate. Coordinate seeding confined
    # to the first 16 entries cannot see the disconnected lowest eigenspace.
    dimension = 64
    physical = np.diag(np.linspace(.25, 3., dimension)).astype(complex)
    physical[-1, -1] = -2.
    if case != 'diagonal':
        rng = np.random.default_rng(190)
        rotation, _ = np.linalg.qr(rng.normal(size=(16, 16))
                                  + 1j*rng.normal(size=(16, 16)))
        physical[-16:, -16:] = rotation @ physical[-16:, -16:] @ rotation.conj().T
    frame = np.eye(dimension, dtype=complex)
    if case == 'singular-scaled':
        # Add both a redundant column and a zero-norm coordinate, then change
        # coefficient scales without changing the physical variational space.
        frame = np.column_stack((frame, frame[:, 0], np.zeros(dimension)))
        frame *= np.geomspace(1e-4, 1e4, frame.shape[1])
    h, n = frame.conj().T@physical@frame, frame.conj().T@frame
    size = frame.shape[1]
    problem = ReducedLocalProblem(0, None,
        SimpleNamespace(source_size=size, target_size=size),
        None if matrix_free else h, None if matrix_free else n,
        hamiltonian_action=lambda x: h@x, metric_action=lambda x: n@x)
    initial = np.zeros(size, dtype=complex)
    initial[0] = 1/frame[0, 0]
    energy, vector, _, _ = _solve_local_problem(problem,
        LETTADMROptions(eigensolver_tolerance=1e-11), initial_vector=initial)
    actual = frame@vector
    expected, states = np.linalg.eigh(physical)
    assert energy == pytest.approx(expected[0], abs=2e-10)
    assert abs(np.vdot(actual, states[:, 0])) == pytest.approx(1., abs=2e-10)
    assert np.linalg.norm(physical@actual-energy*actual) < 2e-9


def test_exploration_preserves_incumbent_inside_degenerate_ground_space():
    dimension = 32
    diagonal = np.r_[np.zeros(20), np.ones(12)]
    problem = ReducedLocalProblem(0, None,
        SimpleNamespace(source_size=dimension, target_size=dimension),
        hamiltonian_action=lambda x: diagonal*x, metric_action=lambda x: x.copy())
    rng = np.random.default_rng(293)
    initial = rng.normal(size=dimension)+1j*rng.normal(size=dimension)
    initial[20:] = 0.
    initial /= np.linalg.norm(initial)
    energy, actual, _, _ = _solve_local_problem(problem, LETTADMROptions(),
                                               initial_vector=initial)
    assert energy == pytest.approx(0., abs=1e-12)
    np.testing.assert_allclose(actual, initial, atol=2e-12)
