"""Bounded, factorized coupled correction after CBE's alternating relaxation.

The dense matrices live in the two factor parameter spaces, never in a merged
pair or global physical space. The caller bounds their combined dimension.
"""
from dataclasses import dataclass

import numpy as np

from .._letta_two_site_opt.energy_refinement import _balance_pair, _spectral_metric_descent
from .contractions import _equilibrated_metric_factors


def _cross_overlap(cache, left_environment, right_environment, layout, left, right):
    """Contract the overlap between variations of A and variations of B."""
    from .cbe import _complete_contraction

    i = layout.left_site
    left_bra, left_ket = cache._group_labels(i)
    right_bra, right_ket = cache._group_labels(i + 1)
    operands = [left_environment, right_environment, left, right.conj()]
    labels = [cache.frontiers[i], cache.frontiers[i + 2], left_ket, right_bra]
    right_output = list(right_ket)
    # A shared physical index is diagonal across the two parameter spaces.
    used = set(label for group in labels for label in group)
    copied = min(used | {0}) - 1
    for axis, label in enumerate(right_output):
        if label in left_bra:
            operands.append(np.eye(right.shape[axis], dtype=left.dtype))
            labels.append((label, copied))
            right_output[axis] = copied
            copied -= 1
    return _complete_contraction(
        operands, labels, tuple(left_bra) + tuple(right_output),
        left.shape + right.shape,
    ).reshape(left.size, right.size)


def _conditioned_direction(metric, gradient, boundary, cutoff, *, overlap=None):
    """Whiten each supported factor block before solving the joint metric."""
    pieces = []
    for selected in (slice(0, boundary), slice(boundary, None)):
        block = metric[selected, selected]
        if not block.size:
            pieces.append(np.zeros((block.shape[0], 0), dtype=metric.dtype))
            continue
        basis, _ = _equilibrated_metric_factors(block, cutoff)
        pieces.append(basis)
    a, b = pieces
    width = a.shape[1] + b.shape[1]
    if not width:
        return np.zeros_like(gradient), 0.0, 0.0
    whitening = np.zeros((gradient.size, width), dtype=metric.dtype)
    whitening[:boundary, :a.shape[1]] = a
    whitening[boundary:, a.shape[1]:] = b
    reduced = whitening.conj().T @ metric @ whitening
    if overlap is not None:
        projected = whitening.conj().T @ overlap
        reduced -= np.outer(projected, projected.conj())
    reduced = 0.5 * (reduced + reduced.conj().T)
    coordinates = whitening.conj().T @ gradient
    direction, residual = _spectral_metric_descent(reduced, coordinates, cutoff)
    # This is the sum of the independent block residuals; reuse the whitening.
    independent = float(np.vdot(coordinates, coordinates).real)
    return whitening @ direction, residual, independent


@dataclass(frozen=True)
class CoupledRefinement:
    left: np.ndarray
    right: np.ndarray
    energy: float
    norm: float
    iterations: int
    accepted_steps: int
    hamiltonian_applications: int


def refine_coupled_factors(
    state, layout, hamiltonian_cache, metric_cache,
    hamiltonian_left, hamiltonian_right, metric_left, metric_right,
    *, options, norm_limit,
):
    """Backtrack simultaneous factor changes using fresh one-site environments.

    Scalar normalization/balancing preserves the physical state. Block whitening
    removes factor scaling before the supported joint natural direction is found.
    Every accepted step must lower the actual streamed energy and respect the
    fixed budget supplied by the caller. Always restore the live input tensors.
    """
    from .cbe import _streamed_bond_energy

    i, j = layout.left_site, layout.left_site + 1
    saved = state.tensors[i], state.tensors[j]
    indices_a = np.flatnonzero(layout.factor_mask('left').reshape(-1))
    indices_b = np.flatnonzero(layout.factor_mask('right').reshape(-1))
    indices = np.concatenate((indices_a, saved[0].size + indices_b))
    checks = accepted = applications = 0

    def install(left, right):
        state.tensors[i], state.tensors[j] = left, right

    def evaluate():
        return _streamed_bond_energy(
            hamiltonian_cache, metric_cache, hamiltonian_left, hamiltonian_right,
            metric_left, metric_right, layout,
        )

    try:
        install(saved[0].copy(), saved[1].copy())
        _, norm = evaluate()
        install(*_balance_pair(state.tensors[i], state.tensors[j] / norm))
        for iteration in range(options.cbe_coupled_max_iterations):
            left, right = state.tensors[i], state.tensors[j]
            energy, norm = evaluate()
            n = norm**2
            na = metric_cache.effective_metric(
                metric_left, metric_cache.extend_right(metric_right, j), i,
            ).to_dense()
            nb = metric_cache.effective_metric(
                metric_cache.extend_left(metric_left, i), metric_right, j,
            ).to_dense()
            cross = _cross_overlap(metric_cache, metric_left, metric_right, layout, left, right)
            overlap = np.concatenate((na @ left.ravel(), nb @ right.ravel())) / n
            metric = np.block([[na, cross], [cross.conj().T, nb]]) / n
            metric = 0.5 * (metric + metric.conj().T)
            ha = hamiltonian_cache.prepare_effective_action(
                hamiltonian_left, hamiltonian_cache.extend_right(hamiltonian_right, j), i,
            )(left.ravel())
            hb = hamiltonian_cache.prepare_effective_action(
                hamiltonian_cache.extend_left(hamiltonian_left, i), hamiltonian_right, j,
            )(right.ravel())
            applications += 2
            gradient = np.concatenate((ha, hb)) / n - energy * overlap
            restricted = metric[np.ix_(indices, indices)]
            delta, residual, independent = _conditioned_direction(
                restricted, gradient[indices], indices_a.size,
                options.cbe_coupled_metric_tolerance,
                overlap=overlap[indices],
            )
            checks += 1
            floor = max(1e-12, np.finfo(metric.real.dtype).eps * indices.size)
            noise = (floor * max(1.0, abs(energy)))**2
            if residual <= noise:
                break
            if iteration == 0 and residual <= options.cbe_coupled_activation_ratio**2 * max(independent, noise):
                break
            full_delta = np.zeros_like(gradient)
            full_delta[indices] = delta
            da = full_delta[:left.size].reshape(left.shape)
            db = full_delta[left.size:].reshape(right.shape)
            slope = float(2 * np.vdot(gradient, full_delta).real)
            if not np.isfinite(slope) or slope >= 0:
                break
            relative = max(np.linalg.norm(da) / np.linalg.norm(left),
                           np.linalg.norm(db) / np.linalg.norm(right))
            step = min(1.0, 0.25 / max(relative, np.finfo(float).tiny))
            good = False
            for trial in range(50):
                install(left + step * da, right + step * db)
                try:
                    _, proposed_norm = evaluate()
                    install(*_balance_pair(state.tensors[i], state.tensors[j] / proposed_norm))
                    proposed, _ = evaluate()
                except (ValueError, FloatingPointError):
                    step *= 0.5
                    continue
                if (max(np.linalg.norm(state.tensors[k]) for k in (i, j)) <= norm_limit
                        and proposed < energy and proposed <= energy + 1e-4 * step * slope):
                    good = True
                    break
                step *= 0.5
            if not good:
                install(left, right)
                break
            accepted += 1
        energy, norm = evaluate()
        return CoupledRefinement(
            state.tensors[i].copy(), state.tensors[j].copy(), energy, norm,
            checks, accepted, applications,
        )
    finally:
        install(*saved)
