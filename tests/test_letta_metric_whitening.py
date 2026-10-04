"""Spectral and adjoint checks for the blockwise two-site metric map."""

import numpy as np
import pytest

from pyqed._letta_one_site_opt.contractions import BlockDiagonalMetric
from pyqed._letta_two_site_opt.solver import _BlockMetricWhitening


@pytest.mark.parametrize("complex_metric", [False, True])
@pytest.mark.parametrize("complex_vector", [False, True])
def test_whitening_isometry_coordinates_and_adjoint(complex_metric, complex_vector):
    rng = np.random.default_rng(91732)
    spectra = [np.array(x) for x in (
        [1.0], [2.0, 0.0], [3.0, 2.0, 1e-12], [2.0, 1.0],
        [1.0, 0.0, 0.0], [4.0, 0.0], [3.0, 1.0, 0.0, 0.0], [0.0] * 4,
    )]
    size = sum(len(x) for x in spectra) + 3
    positions = rng.permutation(size)
    dtype = complex if complex_metric else float
    reference_metric = np.zeros((size, size), dtype=dtype)
    projector = np.zeros_like(reference_metric)
    blocks, indices = [], []
    offset = 0
    rank = 0
    for spectrum in spectra:
        width = len(spectrum)
        matrix = rng.normal(size=(width, width))
        if complex_metric:
            matrix = matrix + 1j * rng.normal(size=matrix.shape)
        basis, _ = np.linalg.qr(matrix)
        selected = positions[offset:offset + width]
        offset += width
        block = (basis * spectrum) @ basis.conj().T
        blocks.append(block)
        indices.append(selected)
        reference_metric[np.ix_(selected, selected)] = block
        retained = spectrum > 4e-10
        projector[np.ix_(selected, selected)] = (
            basis[:, retained] @ basis[:, retained].conj().T
        )
        rank += np.count_nonzero(retained)

    metric = BlockDiagonalMetric(size, blocks, indices)
    whitening = _BlockMetricWhitening(metric, 1e-10)
    assert whitening.rank == rank
    full_storage = rng.normal(size=2 * size)
    reduced_storage = rng.normal(size=2 * rank)
    if complex_vector:
        full_storage = full_storage + 1j * rng.normal(size=full_storage.shape)
        reduced_storage = reduced_storage + 1j * rng.normal(size=reduced_storage.shape)
    full, reduced = full_storage[::2], reduced_storage[::2]
    before_full, before_reduced = full.copy(), reduced.copy()
    identity = np.eye(rank)
    transform = np.column_stack([whitening.to_full(column) for column in identity])
    np.testing.assert_allclose(transform.conj().T @ reference_metric @ transform,
                               identity, atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(whitening.to_full(reduced), transform @ reduced,
                               atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(whitening.adjoint(full), transform.conj().T @ full,
                               atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(whitening.coordinates(whitening.to_full(reduced)),
                               reduced, atol=3e-14, rtol=3e-14)
    # Equilibration changes the representative in the null space, not the
    # physical state. A Euclidean projector is not required in these coordinates.
    reconstructed = whitening.to_full(whitening.coordinates(full))
    np.testing.assert_allclose(reference_metric @ reconstructed,
                               reference_metric @ (projector @ full), atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(whitening.coordinates(full),
                               transform.conj().T @ reference_metric @ full,
                               atol=2e-11, rtol=2e-11)
    np.testing.assert_array_equal(whitening.to_full(reduced)[positions[offset:]], 0)
    np.testing.assert_array_equal(full, before_full)
    np.testing.assert_array_equal(reduced, before_reduced)
    assert whitening.to_full(reduced).dtype == np.result_type(metric.dtype, reduced)
    assert whitening.adjoint(full).dtype == np.result_type(metric.dtype, full)
    assert whitening.coordinates(full).dtype == np.result_type(metric.dtype, full)


def test_whitening_rejects_zero_rank_metric():
    metric = BlockDiagonalMetric(3, [np.zeros((3, 3))], [np.arange(3)])
    with pytest.raises(ValueError, match="zero rank"):
        _BlockMetricWhitening(metric, 1e-10)


@pytest.mark.parametrize("input_dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_whitening_preserves_mixed_block_arithmetic(input_dtype):
    dtypes = [np.float32, np.float64, np.complex64, np.complex128]
    blocks = [np.diag(np.array([1.0, 4.0], dtype=dtype)) for dtype in dtypes]
    metric = BlockDiagonalMetric(8, blocks, [np.arange(i, i + 2) for i in range(0, 8, 2)])
    whitening = _BlockMetricWhitening(metric, 1e-10)
    vector = np.random.default_rng(8712).normal(size=8).astype(input_dtype)
    if np.iscomplexobj(vector):
        vector += 0.73j * vector
    expected = np.empty(8, dtype=np.complex128)
    expected_coordinates = np.empty_like(expected)
    for i, dtype in enumerate(dtypes):
        selected = slice(2 * i, 2 * i + 2)
        roots = np.array([1.0, 2.0], dtype=np.empty((), dtype=dtype).real.dtype)
        basis = np.eye(2, dtype=dtype)
        expected[selected] = basis @ (vector[selected] / roots)
        expected_coordinates[selected] = roots * (basis.conj().T @ vector[selected])
    np.testing.assert_array_equal(whitening.to_full(vector), expected)
    np.testing.assert_array_equal(whitening.adjoint(vector), expected)
    np.testing.assert_array_equal(whitening.coordinates(vector), expected_coordinates)
