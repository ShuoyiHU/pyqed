"""Experimental Pippan--White--Evertz style LETTA segment compression.

A segment is a linear map between two *frontier tensors*. Its singular
vectors are not Schmidt vectors and its rank is not the MPS bond dimension.
Snapshots remain valid after the state changes, but must be rebuilt to
represent the changed state. Production sweeps do not select this path yet.
"""
from dataclasses import dataclass
from operator import index

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .contractions import _contract_operands


def _positive_integer(value, name, *, allow_zero=False):
    value = index(value)
    if value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}.")
    return value


class SegmentTransfer(LinearOperator):
    """Frozen exact map for sites ``[start, stop)`` of an environment cache.

    Rows are flattened frontier indices at ``stop``; columns are those at
    ``start``. Keeping labels that cross several sites preserves LETTA's
    physical copy constraints, including nonlocal/backward ties. No dense
    segment matrix or many-body wavefunction is constructed.

    Both ordinary and pair caches are supported, for norm and MPO networks.
    The source cache must use exact boundaries. Snapshots deliberately own
    their tensors: later in-place edits, gauge changes and bond expansions
    cannot silently invalidate the operator.
    """

    def __init__(self, cache, start, stop):
        start, stop = index(start), index(stop)
        if not 0 <= start < stop <= cache.state.nsites:
            raise ValueError("require 0 <= start < stop <= number of sites.")
        if cache.boundary_bond_dim is not None:
            raise ValueError("segment transfers require exact source boundaries.")
        self.start, self.stop = start, stop
        self.frontiers = tuple(cache.frontiers[start:stop + 1])
        self.frontier_shapes = tuple(
            tuple(cache.label_dimensions[label] for label in labels)
            for labels in self.frontiers
        )
        self.input_shape, self.output_shape = self.frontier_shapes[0], self.frontier_shapes[-1]
        self.groups = []
        for site in range(start, stop):
            operands, labels = cache._group(site)
            frozen = tuple(np.array(value, copy=True) for value in operands)
            for value in frozen:
                value.flags.writeable = False
            self.groups.append((frozen, tuple(map(tuple, labels))))
        self.groups = tuple(self.groups)
        dtype = np.result_type(*(a.dtype for operands, _ in self.groups for a in operands))
        super().__init__(dtype=dtype, shape=(int(np.prod(self.output_shape)),
                                            int(np.prod(self.input_shape))))

    def _matvec(self, vector):
        boundary = np.asarray(vector).reshape(self.input_shape)
        for cut, (operands, labels) in enumerate(self.groups):
            boundary = _contract_operands(
                [boundary, *operands], [self.frontiers[cut], *labels], self.frontiers[cut + 1])
        return np.asarray(boundary).reshape(-1)

    def _rmatvec(self, vector):
        # Right environment propagation is M.T, whereas scipy requires M.H.
        boundary = np.asarray(vector).conj().reshape(self.output_shape)
        for cut in reversed(range(len(self.groups))):
            operands, labels = self.groups[cut]
            boundary = _contract_operands(
                [boundary, *operands], [self.frontiers[cut + 1], *labels], self.frontiers[cut])
        return np.asarray(boundary).conj().reshape(-1)

    def apply_left(self, boundary):
        """Propagate a left environment from start to stop."""
        return self.matvec(np.asarray(boundary).reshape(-1)).reshape(self.output_shape)

    def apply_right(self, boundary):
        """Propagate a right environment from stop to start (transpose)."""
        vector = np.asarray(boundary).reshape(-1)
        return self.rmatvec(vector.conj()).conj().reshape(self.input_shape)

    def compress(self, rank, *, oversampling=8, power_iterations=1,
                 tolerance=1e-8, validation_vectors=8, seed=0,
                 max_workspace_mb=128.0):
        """Randomized row-space SVD, with independent two-sided probes.

        A failed error/storage check returns an object whose propagation
        methods use the exact snapshot. Probe errors estimate action error;
        they do NOT certify positivity, an operator-norm bound or energies.
        ``candidate`` exposes the unguarded low-rank map for diagnostics.
        ``max_workspace_mb`` bounds estimated sketch/SVD storage; contraction
        intermediates and the already-owned snapshot are outside this budget.
        """
        rank = min(_positive_integer(rank, "rank"), min(self.shape))
        oversampling = _positive_integer(oversampling, "oversampling", allow_zero=True)
        power_iterations = _positive_integer(power_iterations, "power_iterations", allow_zero=True)
        validation_vectors = _positive_integer(validation_vectors, "validation_vectors")
        if not np.isfinite(tolerance) or not 0 <= tolerance < 1:
            raise ValueError("tolerance must be finite and in [0, 1).")
        if not np.isfinite(max_workspace_mb) or max_workspace_mb <= 0:
            raise ValueError("max_workspace_mb must be finite and positive.")
        m, n = self.shape
        width = min(rank + oversampling, m, n)
        # QR, skinny SVD and their working copies; probes are streamed.
        itemsize = np.dtype(np.result_type(self.dtype, np.float64)).itemsize
        workspace = itemsize * (8 * (m + n) * width + 8 * width**2)
        if workspace > max_workspace_mb * 1024**2:
            raise MemoryError(f"segment SVD estimated workspace is {workspace / 1024**2:.2f} MiB.")
        sketch_seed, check_seed = np.random.SeedSequence(seed).spawn(2)
        rng = np.random.default_rng(sketch_seed)

        def random(shape, generator):
            values = generator.normal(size=shape)
            if np.issubdtype(self.dtype, np.complexfloating):
                values = (values + 1j * generator.normal(size=shape)) / np.sqrt(2.)
            return values

        q, _ = np.linalg.qr(self.H @ random((m, width), rng), mode="reduced")
        for _ in range(power_iterations):
            p, _ = np.linalg.qr(self @ q, mode="reduced")
            q, _ = np.linalg.qr(self.H @ p, mode="reduced")
        u, singular_values, wh = np.linalg.svd(self @ q, full_matrices=False)
        candidate = LowRankTransfer(u[:, :rank], singular_values[:rank], wh[:rank] @ q.conj().T)
        rng = np.random.default_rng(check_seed)
        errors = []
        for exact, approximation in ((self, candidate), (self.H, candidate.H)):
            numerator, denominator = 0., 0.
            for _ in range(validation_vectors):
                probe = random((exact.shape[1],), rng)
                expected = exact @ probe
                delta = expected - approximation @ probe
                numerator += np.vdot(delta, delta).real
                denominator += np.vdot(expected, expected).real
            errors.append(float(np.sqrt(numerator / denominator)) if denominator else
                          (0. if numerator == 0 else float("inf")))
        storage_ratio = candidate.nbytes / (m * n * itemsize)
        accepted = max(errors) <= tolerance and storage_ratio < 1.
        reason = ("accepted" if accepted else "probe error exceeds tolerance"
                  if max(errors) > tolerance else "factors do not save dense-map storage")
        return CompressedSegmentTransfer(self, candidate, accepted, {
            "shape": self.shape, "rank": rank, "sketch_width": width,
            "forward_probe_error": errors[0], "adjoint_probe_error": errors[1],
            "tolerance": tolerance, "factor_to_dense_storage_ratio": storage_ratio,
            "estimated_workspace_bytes": workspace, "accepted": accepted, "reason": reason,
        })


class LowRankTransfer(LinearOperator):
    """Matrix-free U diag(s) Vh with an actual complex adjoint."""

    def __init__(self, u, singular_values, vh):
        self.u, self.singular_values, self.vh = u, singular_values, vh
        super().__init__(dtype=np.result_type(u, vh), shape=(u.shape[0], vh.shape[1]))

    @property
    def nbytes(self):
        return self.u.nbytes + self.singular_values.nbytes + self.vh.nbytes

    def _matvec(self, vector):
        return self.u @ (self.singular_values * (self.vh @ np.asarray(vector).reshape(-1)))

    def _rmatvec(self, vector):
        return self.vh.conj().T @ (self.singular_values * (self.u.conj().T @ np.asarray(vector).reshape(-1)))


@dataclass(frozen=True)
class CompressedSegmentTransfer:
    exact: SegmentTransfer
    candidate: LowRankTransfer
    accepted: bool
    diagnostics: dict

    @property
    def operator(self):
        return self.candidate if self.accepted else self.exact

    def apply_left(self, boundary):
        return self.operator.matvec(np.asarray(boundary).reshape(-1)).reshape(self.exact.output_shape)

    def apply_right(self, boundary):
        vector = np.asarray(boundary).reshape(-1)
        return self.operator.rmatvec(vector.conj()).conj().reshape(self.exact.input_shape)
