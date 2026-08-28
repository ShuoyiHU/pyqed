import numpy as np
import pytest
import pyqed._letta_one_site_opt.reduced_solver as reduced_solver_module

from pyqed._letta_one_site_opt import (
    LETTADMROptions,
    ReducedFrontier,
    ReducedLatticeLETTA,
    ReducedPhysicalBasis,
    ReducedSymmetry,
    letta_dmrg,
    reduced_local_problem,
    su2_heisenberg_mpo,
)


def _heisenberg_dense(nsites, coupling=1.0):
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = 0.5 * np.diag([1.0, -1.0])
    identity = np.eye(2)
    hamiltonian = np.zeros((2**nsites, 2**nsites), dtype=complex)
    for site in range(nsites - 1):
        for operator in (sx, sy, sz):
            factors = [identity] * nsites
            factors[site] = operator
            factors[site + 1] = operator
            term = factors[0]
            for factor in factors[1:]:
                term = np.kron(term, factor)
            hamiltonian += coupling * term
    return hamiltonian


def _random_singlet_state(nsites=4, seed=5):
    symmetry = ReducedSymmetry.su2(
        ReducedPhysicalBasis.spin_half(),
        target_two_j=0,
    )
    return ReducedLatticeLETTA.random(
        (1, nsites),
        symmetry=symmetry,
        multiplets_per_sector=1,
        seed=seed,
    )


def test_reduced_local_problem_matches_explicit_dense_frame():
    state = _random_singlet_state(seed=7)
    hamiltonian = _heisenberg_dense(state.nsites)

    problem = reduced_local_problem(state, hamiltonian, site=1)
    frame = problem.frame

    np.testing.assert_allclose(problem.hamiltonian, frame.conj().T @ hamiltonian @ frame)
    np.testing.assert_allclose(problem.metric, frame.conj().T @ frame)
    assert problem.local_dimension == problem.embedding.source_size
    assert problem.full_local_dimension == problem.embedding.target_size
    # A pure spin-1/2 site has one reduced physical label.  Consequently the
    # LETTA scalar-conditioning coordinate and the frontier coordinate are
    # identical here; the saving is against magnetic-component tensors, not
    # against this already-reduced frontier representation.
    assert problem.local_dimension == problem.full_local_dimension


def test_one_site_reduced_sweep_never_leaves_target_singlet_sector():
    state = _random_singlet_state(seed=11)
    result = letta_dmrg(
        _heisenberg_dense(state.nsites),
        state=state,
        options=LETTADMROptions(
            max_sweeps=4,
            tolerance=1.0e-11,
            gauge_mode="none",
        ),
    )

    assert result.state.target_two_j == 0
    assert result.state.symmetry_violation() == pytest.approx(0.0)
    assert all(
        later.energy <= earlier.energy + 1.0e-10
        for earlier, later in zip(result.history, result.history[1:])
    )


def test_one_site_reduced_heisenberg_energy_matches_exact_singlet_ground_state():
    state = _random_singlet_state(seed=17)
    hamiltonian = _heisenberg_dense(state.nsites)
    exact = np.linalg.eigvalsh(hamiltonian)[0]

    result = letta_dmrg(
        hamiltonian,
        state=state,
        options=LETTADMROptions(
            max_sweeps=12,
            tolerance=1.0e-12,
            metric_tolerance=1.0e-13,
            gauge_mode="none",
        ),
    )

    assert result.energy == pytest.approx(exact, abs=1.0e-10)
    assert result.history[-1].updates
    assert all(
        update.local_dimension <= update.full_local_dimension
        for sweep in result.history
        for update in sweep.updates
    )


def test_reduced_local_frame_agrees_with_frontier_parameter_embedding():
    state = _random_singlet_state(seed=23)
    problem = reduced_local_problem(state, _heisenberg_dense(state.nsites), site=2)
    source = problem.embedding.pack_source(state.tensors[2])
    expanded = problem.embedding.apply(source)
    mps_site = ReducedFrontier.from_state(state).to_mps(state)[2]

    np.testing.assert_allclose(expanded, problem.embedding.pack_target(mps_site.data))
    np.testing.assert_allclose(problem.frame @ source, state.state_vector())


def test_public_dispatch_constructs_user_selected_reduced_sector():
    basis = ReducedPhysicalBasis.spin_half()
    symmetry = ReducedSymmetry.su2(basis, target_two_j=0)
    result = letta_dmrg(
        su2_heisenberg_mpo(4, physical_basis=basis),
        lattice_shape=(1, 4),
        bond_dim=1,
        seed=59,
        symmetry=symmetry,
        options=LETTADMROptions(
            max_sweeps=4,
            tolerance=1.0e-10,
            gauge_mode="none",
        ),
    )

    assert isinstance(result.state, ReducedLatticeLETTA)
    assert result.state.symmetry == symmetry
    assert result.state.symmetry_violation() == 0.0


def test_canonical_matrix_free_local_actions_match_dense_projection():
    state = _random_singlet_state(seed=67)
    hamiltonian = su2_heisenberg_mpo(
        state.nsites, physical_basis=state.physical_basis
    )
    dense = reduced_local_problem(state, hamiltonian, site=1)
    matrix_free = reduced_local_problem(
        state,
        hamiltonian,
        site=1,
        matrix_free=True,
        dense_solver_threshold=1,
    )
    rng = np.random.default_rng(71)
    vector = rng.normal(size=dense.local_dimension) + 1.0j * rng.normal(
        size=dense.local_dimension
    )

    assert matrix_free.hamiltonian is None
    assert matrix_free.metric is None
    np.testing.assert_allclose(
        matrix_free.hamiltonian_action(vector),
        dense.hamiltonian @ vector,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        matrix_free.metric_action(vector),
        dense.metric @ vector,
        atol=2.0e-12,
    )


def test_reduced_one_site_matrix_free_sweep_avoids_dense_source_frame(monkeypatch):
    state = _random_singlet_state(seed=73)
    hamiltonian = su2_heisenberg_mpo(
        state.nsites, physical_basis=state.physical_basis
    )
    reference = letta_dmrg(
        hamiltonian,
        state=state,
        options=LETTADMROptions(
            max_sweeps=1,
            matrix_free=False,
            gauge_mode="none",
        ),
    )

    original = reduced_solver_module._expanded_source_frame

    def forbidden(sites, embedding, site):
        if embedding.source_size > 1:
            raise AssertionError("matrix-free reduced solve built a dense source frame")
        return original(sites, embedding, site)

    monkeypatch.setattr(reduced_solver_module, "_expanded_source_frame", forbidden)
    result = letta_dmrg(
        hamiltonian,
        state=state,
        options=LETTADMROptions(
            max_sweeps=1,
            matrix_free=True,
            dense_solver_threshold=1,
            gauge_mode="none",
        ),
    )

    assert np.isfinite(result.energy)
    assert result.state.norm() == pytest.approx(1.0, abs=1.0e-10)
    assert result.energy == pytest.approx(reference.energy, abs=2.0e-10)
