import numpy as np
import pytest

from pyqed._letta_one_site_opt import (
    ReducedLatticeLETTA,
    ReducedPhysicalBasis,
    ReducedSymmetry,
)
from pyqed.mps.su2 import SpinChargeSector, SU2Irrep


def _spin_half_singlet_symmetry():
    return ReducedSymmetry.su2(
        ReducedPhysicalBasis.spin_half(),
        target_two_j=0,
    )


def test_random_reduced_letta_stores_only_fusion_allowed_blocks():
    symmetry = _spin_half_singlet_symmetry()
    state = ReducedLatticeLETTA.random(
        (1, 2),
        symmetry=symmetry,
        multiplets_per_sector=1,
        seed=3,
    )

    vacuum = SpinChargeSector(0, SU2Irrep(0))
    half = SpinChargeSector(0, SU2Irrep(1))
    assert tuple(state.tensors[0]) == ((vacuum, half, half),)
    assert tuple(state.tensors[1]) == ((half, half, vacuum),)
    assert state.tensors[0][(vacuum, half, half)].shape == (1, 1, 1, 1)
    assert state.tensors[1][(half, half, vacuum)].shape == (1, 1, 1)
    assert state.parameter_count == 2
    assert state.symmetry_violation() == pytest.approx(0.0)


def test_reduced_letta_rejects_forbidden_fusion_block():
    symmetry = _spin_half_singlet_symmetry()
    vacuum = symmetry.identity
    half = symmetry.physical_basis.sectors[0]
    forbidden = {
        (vacuum, half, vacuum): np.ones((1, 1, 1, 1)),
    }
    second = {
        (half, half, vacuum): np.ones((1, 1, 1)),
    }

    with pytest.raises(ValueError, match="forbidden fusion block"):
        ReducedLatticeLETTA(
            (1, 2),
            symmetry,
            (forbidden, second),
            bond_sectors=((half,),),
            normalize=False,
        )


def test_two_spin_reduced_letta_expands_to_exact_singlet():
    symmetry = _spin_half_singlet_symmetry()
    vacuum = symmetry.identity
    half = symmetry.physical_basis.sectors[0]
    state = ReducedLatticeLETTA(
        (1, 2),
        symmetry,
        (
            {(vacuum, half, half): np.ones((1, 1, 1, 1))},
            {(half, half, vacuum): np.ones((1, 1, 1))},
        ),
        bond_sectors=((half,),),
    )

    expected = np.array([0.0, 1.0, -1.0, 0.0]) / np.sqrt(2.0)
    np.testing.assert_allclose(state.state_vector(), expected, atol=1.0e-13)

    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = 0.5 * np.diag([1.0, -1.0])
    total_s2 = sum(
        total_op @ total_op
        for total_op in (
            np.kron(op, np.eye(2)) + np.kron(np.eye(2), op)
            for op in (sx, sy, sz)
        )
    )
    assert np.vdot(state.state_vector(), total_s2 @ state.state_vector()).real == pytest.approx(
        0.0, abs=1.0e-13
    )


def test_copy_and_normalize_preserve_exact_target_sector():
    state = ReducedLatticeLETTA.random(
        (1, 4),
        symmetry=_spin_half_singlet_symmetry(),
        multiplets_per_sector=1,
        seed=11,
    )
    copied = state.copy()

    np.testing.assert_allclose(copied.state_vector(), state.state_vector())
    assert np.linalg.norm(copied.state_vector()) == pytest.approx(1.0)
    assert copied.target_two_j == 0
    assert copied.symmetry_violation() == pytest.approx(0.0)


def test_spatial_orbital_conditioning_axes_use_reduced_dimension_not_dense_dimension():
    basis = ReducedPhysicalBasis.spatial_orbital()
    symmetry = ReducedSymmetry.su2(
        basis,
        target_charge=2,
        target_two_j=0,
    )
    state = ReducedLatticeLETTA.random(
        (1, 2),
        symmetry=symmetry,
        multiplets_per_sector=1,
        seed=5,
    )

    first_shapes = [block.shape for block in state.tensors[0].values()]
    assert first_shapes
    assert all(shape[2] == basis.reduced_dim == 3 for shape in first_shapes)
    assert basis.dense_dim == 4


def test_reduced_dense_reconstruction_selects_requested_target_component():
    symmetry = ReducedSymmetry.su2(
        ReducedPhysicalBasis.spin_half(),
        target_two_j=1,
    )
    state = ReducedLatticeLETTA.random(
        (1, 3),
        symmetry=symmetry,
        multiplets_per_sector=1,
        seed=7,
    )

    plus = state.state_vector(target_two_m=1)
    minus = state.state_vector(target_two_m=-1)
    assert np.linalg.norm(plus) == pytest.approx(1.0)
    assert np.linalg.norm(minus) == pytest.approx(1.0)
    assert np.vdot(plus, minus) == pytest.approx(0.0, abs=1.0e-12)

    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = 0.5 * np.diag([1.0, -1.0])
    identity = np.eye(2)
    total_s2 = np.zeros((8, 8), dtype=complex)
    for local in (sx, sy, sz):
        total = sum(
            np.kron(np.kron(local if site == 0 else identity,
                              local if site == 1 else identity),
                    local if site == 2 else identity)
            for site in range(3)
        )
        total_s2 += total @ total
    target_s2 = 0.5 * (0.5 + 1.0)
    np.testing.assert_allclose(total_s2 @ plus, target_s2 * plus, atol=1.0e-12)
    np.testing.assert_allclose(total_s2 @ minus, target_s2 * minus, atol=1.0e-12)


def test_large_reduced_state_normalization_never_needs_dense_hilbert_space():
    state = ReducedLatticeLETTA.random(
        (1, 16),
        symmetry=_spin_half_singlet_symmetry(),
        multiplets_per_sector=1,
        seed=29,
    )

    assert state.norm() == pytest.approx(1.0, abs=1.0e-12)
    with pytest.raises(ValueError, match="verification-only"):
        state.state_vector()


def test_scalar_gauge_balance_preserves_state_and_removes_scale_imbalance():
    state = ReducedLatticeLETTA.random(
        (1, 4),
        symmetry=_spin_half_singlet_symmetry(),
        multiplets_per_sector=2,
        seed=41,
    )
    for key in state.tensors[0]:
        state.tensors[0][key] *= 1.0e8
    for key in state.tensors[1]:
        state.tensors[1][key] *= 1.0e-8
    before = state.state_vector()

    state.balance_scalar_gauge()

    after = state.state_vector()
    norms = [
        np.sqrt(sum(np.vdot(block, block).real for block in tensor.values()))
        for tensor in state.tensors
    ]
    np.testing.assert_allclose(after, before, atol=1.0e-12, rtol=1.0e-12)
    assert max(norms) / min(norms) < 1.0 + 1.0e-12
