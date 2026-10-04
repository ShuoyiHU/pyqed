"""Independent spectral checks for the overlap action used by metric ALS."""

import numpy as np
import pytest

from pyqed._letta_one_site_opt.contractions import BlockDiagonalMetric
from pyqed._letta_two_site_opt.truncation import _MetricSquareRoot


@pytest.mark.parametrize("complex_metric", [False, True])
@pytest.mark.parametrize("complex_vector", [False, True])
def test_metric_factor_preserves_known_physical_norm_and_adjoint(complex_metric, complex_vector):
    rng = np.random.default_rng(17091)
    spectra = [np.array(x) for x in (
        [1.0], [2.0, 0.0], [3.0, 2.0, 1e-12], [2.0, 1.0],
        [1.0, 0.0, 0.0], [4.0, 0.0], [3.0, 1.0, 0.0, 0.0], [0.0] * 4,
    )]
    size = sum(len(x) for x in spectra) + 3
    positions = rng.permutation(size)
    blocks, indices = [], []
    reference = np.zeros((size, size), dtype=complex if complex_metric else float)
    offset = 0
    for spectrum in spectra:
        width = len(spectrum)
        matrix = rng.normal(size=(width, width))
        if complex_metric:
            matrix = matrix + 1j * rng.normal(size=matrix.shape)
        basis, _ = np.linalg.qr(matrix)
        selected = positions[offset:offset + width]
        offset += width
        blocks.append((basis * spectrum) @ basis.conj().T)
        indices.append(selected)
        # The nonzero spectrum is separated from the existing 1e-10 cutoff.
        roots = np.sqrt(np.where(spectrum > 4e-10, spectrum, 0.0))
        reference[np.ix_(selected, selected)] = (basis * roots) @ basis.conj().T
    metric = BlockDiagonalMetric(size, blocks, indices)
    root = _MetricSquareRoot(metric, 1e-10)
    vector = rng.normal(size=size * 2)[::2]
    if complex_vector:
        storage = rng.normal(size=size * 2) + 1j * rng.normal(size=size * 2)
        vector = storage[::2]
    before = vector.copy()
    actual = root.apply(vector)
    # Any S with S^H S = N is a valid least-squares factor; S itself need not
    # equal the Hermitian principal square root. Weak dependent directions
    # may differ at the explicitly requested rank tolerance.
    np.testing.assert_allclose(np.vdot(actual, actual),
                               np.vdot(reference @ vector, reference @ vector),
                               rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(root.adjoint(actual), reference @ reference @ vector,
                               rtol=2e-11, atol=2e-11)
    probe = rng.normal(size=size) + (1j * rng.normal(size=size) if complex_vector else 0.)
    np.testing.assert_allclose(np.vdot(probe, root.apply(vector)),
                               np.vdot(root.adjoint(probe), vector), atol=3e-14, rtol=3e-14)
    np.testing.assert_array_equal(actual[positions[offset:]], 0)
    np.testing.assert_array_equal(vector, before)
    assert actual.dtype == np.result_type(metric.dtype, vector)


def test_zero_rank_metric_is_rejected():
    metric = BlockDiagonalMetric(3, [np.zeros((3, 3))], [np.arange(3)])
    with pytest.raises(ValueError, match="zero rank"):
        _MetricSquareRoot(metric, 1e-10)


@pytest.mark.parametrize("input_dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_mixed_block_dtypes_preserve_metric_norm(input_dtype):
    rng = np.random.default_rng(3813)
    blocks = []
    for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
        factor = rng.normal(size=(3, 3)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            factor += 1j * rng.normal(size=factor.shape)
        blocks.append(factor @ factor.conj().T)
    indices = [np.arange(i, i + 3) for i in range(0, 12, 3)]
    metric = BlockDiagonalMetric(12, blocks, indices)
    vector = rng.normal(size=12).astype(input_dtype)
    if np.issubdtype(input_dtype, np.complexfloating):
        vector += 1j * rng.normal(size=vector.shape)
    root = _MetricSquareRoot(metric, 1e-10)
    actual = root.apply(vector)
    np.testing.assert_allclose(np.vdot(actual, actual), np.vdot(vector, metric @ vector),
                               atol=2e-5, rtol=2e-6)
    np.testing.assert_allclose(root.adjoint(actual), metric @ vector, atol=2e-5, rtol=2e-6)
    assert actual.dtype == np.result_type(metric.dtype, vector)
