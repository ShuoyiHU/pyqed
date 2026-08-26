"""JAX gradients for finite-window uniform plane LETTA contractions."""

from __future__ import annotations

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from .plane_environment import PlaneEnvironmentOptions
from .plane_operators import (
    PlaneTFIM,
    UnreliablePlaneEnvironmentError,
)


@jax.custom_jvp
def _stable_svd(matrix):
    """Reduced SVD with a gauge-stable derivative at degeneracies."""

    return jnp.linalg.svd(matrix, full_matrices=False)


@_stable_svd.defjvp
def _stable_svd_jvp(primals, tangents):
    (matrix,), (matrix_tangent,) = primals, tangents
    u, singular_values, vh = _stable_svd(matrix)
    v = vh.T
    projected = u.T @ matrix_tangent @ v
    singular_tangent = jnp.diag(projected)

    row_values = singular_values[:, None]
    column_values = singular_values[None, :]
    squared_differences = (
        (column_values + row_values) * (column_values - row_values)
    )
    scale = jnp.maximum(jnp.max(singular_values**2), 1.0)
    distinct = jnp.abs(squared_differences) > 1.0e-12 * scale
    safe_differences = jnp.where(distinct, squared_differences, 1.0)
    inverse_differences = jnp.where(
        distinct,
        1.0 / safe_differences,
        0.0,
    )

    largest = jnp.max(singular_values)
    nonzero = singular_values > 1.0e-12 * largest
    safe_singular_values = jnp.where(nonzero, singular_values, 1.0)
    inverse_singular_values = jnp.where(
        nonzero,
        1.0 / safe_singular_values,
        0.0,
    )
    inverse_singular_matrix = jnp.diag(inverse_singular_values)

    right_scaled = column_values * projected
    left_scaled = row_values * projected
    skew_diagonal = (
        0.5
        * (projected - projected.T)
        * inverse_singular_matrix
    )
    u_tangent = u @ (
        inverse_differences * (right_scaled + right_scaled.T)
        + skew_diagonal
    )
    v_tangent = v @ (
        inverse_differences * (left_scaled + left_scaled.T)
    )

    rows, columns = matrix.shape
    if rows > columns:
        projected_right = matrix_tangent @ v
        residual = projected_right - u @ (u.T @ projected_right)
        u_tangent = (
            u_tangent
            + residual * inverse_singular_values[None, :]
        )
    if columns > rows:
        projected_left = matrix_tangent.T @ u
        residual = projected_left - v @ (v.T @ projected_left)
        v_tangent = (
            v_tangent
            + residual * inverse_singular_values[None, :]
        )
    return (
        (u, singular_values, vh),
        (u_tangent, singular_tangent, v_tangent.T),
    )


def _double_layer_cell(tensor, operator):
    tensor = tensor / jnp.linalg.norm(tensor)
    local_dim = tensor.shape[4]
    identity = jnp.eye(local_dim, dtype=tensor.dtype)
    cell = jnp.einsum(
        "LRUDCXY,lrudcxy,Cc,CP,cp->LlCcRrXxUuPpDdYy",
        tensor,
        tensor,
        operator,
        identity,
        identity,
        optimize=True,
    )
    dimension = tensor.shape[0] ** 2 * local_dim**2
    return cell.reshape((dimension,) * 4)


def _boundary_vector(bond_dim, local_dim, dtype):
    vector = jnp.einsum(
        "ab,ij->abij",
        jnp.eye(bond_dim, dtype=dtype),
        jnp.eye(local_dim, dtype=dtype),
        optimize=True,
    ).reshape(-1)
    return vector / jnp.linalg.norm(vector)


def _mps_norm(cores):
    environment = jnp.ones((1, 1), dtype=cores[0].dtype)
    for core in cores:
        environment = jnp.einsum(
            "aA,apb,ApB->bB",
            environment,
            core,
            core,
            optimize=True,
        )
    return jnp.sqrt(jnp.maximum(environment.reshape(()), 0.0))


def _compress_mps(cores, max_bond_dim, cutoff):
    cores = list(cores)
    for site in range(len(cores) - 1):
        left, physical, right = cores[site].shape
        matrix = cores[site].reshape(left * physical, right)
        u, singular_values, vh = _stable_svd(matrix)
        keep = min(max_bond_dim, singular_values.shape[0])
        threshold = cutoff * singular_values[0]
        retained = jnp.where(
            singular_values[:keep] > threshold,
            singular_values[:keep],
            0.0,
        )
        cores[site] = u[:, :keep].reshape(left, physical, keep)
        transfer = retained[:, None] * vh[:keep]
        cores[site + 1] = jnp.tensordot(
            transfer,
            cores[site + 1],
            axes=([1], [0]),
        )
    return cores


def _apply_row(cores, row_cells, boundary, max_bond_dim, cutoff):
    updated = []
    last_column = len(cores) - 1
    for column, (core, cell) in enumerate(zip(cores, row_cells)):
        applied = jnp.einsum(
            "aub,lrud->aldbr",
            core,
            cell,
            optimize=True,
        )
        if column == 0:
            applied = jnp.einsum(
                "aldbr,l->adbr",
                applied,
                boundary,
                optimize=True,
            )
        if column == last_column:
            if column == 0:
                applied = jnp.einsum(
                    "adbr,r->adb",
                    applied,
                    boundary,
                    optimize=True,
                )
            else:
                applied = jnp.einsum(
                    "aldbr,r->aldb",
                    applied,
                    boundary,
                    optimize=True,
                )
        if column == 0 and column == last_column:
            new_core = applied
        elif column == 0:
            new_core = applied.reshape(
                applied.shape[0],
                applied.shape[1],
                applied.shape[2] * applied.shape[3],
            )
        elif column == last_column:
            new_core = applied.reshape(
                applied.shape[0] * applied.shape[1],
                applied.shape[2],
                applied.shape[3],
            )
        else:
            new_core = applied.reshape(
                applied.shape[0] * applied.shape[1],
                applied.shape[2],
                applied.shape[3] * applied.shape[4],
            )
        updated.append(new_core)
    return _compress_mps(updated, max_bond_dim, cutoff)


def _contract_window(
    base_cell,
    replacements,
    size,
    bond_dim,
    local_dim,
    boundary_bond_dim,
    cutoff,
):
    boundary = _boundary_vector(
        bond_dim,
        local_dim,
        base_cell.dtype,
    )
    cores = [
        boundary.reshape(1, boundary.size, 1)
        for _ in range(size)
    ]
    log_scale = jnp.asarray(0.0, dtype=base_cell.dtype)
    for row in range(size):
        row_cells = [
            replacements.get((row, column), base_cell)
            for column in range(size)
        ]
        cores = _apply_row(
            cores,
            row_cells,
            boundary,
            boundary_bond_dim,
            cutoff,
        )
        norm = _mps_norm(cores)
        cores[0] = cores[0] / norm
        log_scale = log_scale + jnp.log(norm)

    value = jnp.ones((1,), dtype=base_cell.dtype)
    for core in cores:
        matrix = jnp.einsum(
            "apb,p->ab",
            core,
            boundary,
            optimize=True,
        )
        value = value @ matrix
    return value.reshape(()), log_scale


def _make_energy(shape, model, environment):
    size = environment.window_sizes[-1]
    bond_dim = shape[0]
    local_dim = shape[4]
    center = size // 2
    identity = jnp.eye(local_dim)
    x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]])
    z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]])

    def energy(parameters):
        tensor = parameters.reshape(shape)
        tensor = tensor / jnp.linalg.norm(tensor)
        identity_cell = _double_layer_cell(tensor, identity)
        x_cell = _double_layer_cell(tensor, x)
        z_cell = _double_layer_cell(tensor, z)
        contraction_arguments = (
            size,
            bond_dim,
            local_dim,
            environment.boundary_bond_dim,
            environment.cutoff,
        )
        norm_value, norm_log = _contract_window(
            identity_cell,
            {},
            *contraction_arguments,
        )

        def ratio(replacements):
            value, log_scale = _contract_window(
                identity_cell,
                replacements,
                *contraction_arguments,
            )
            return (
                value
                / norm_value
                * jnp.exp(log_scale - norm_log)
            )

        transverse = ratio({(center, center): x_cell})
        horizontal = ratio(
            {
                (center, center): z_cell,
                (center, center + 1): z_cell,
            }
        )
        vertical = ratio(
            {
                (center, center): z_cell,
                (center + 1, center): z_cell,
            }
        )
        value = (
            -model.coupling * (horizontal + vertical)
            - model.field * transverse
        )
        return value, jnp.stack((transverse, horizontal, vertical))

    return energy


def make_plane_energy_value_and_gradient(shape, model, environment):
    """Compile a real finite-window energy and reverse-mode gradient."""

    shape = tuple(int(dimension) for dimension in shape)
    if (
        len(shape) != 7
        or len(set(shape[:4])) != 1
        or len(set(shape[4:])) != 1
    ):
        raise ValueError("shape must describe a uniform plane LETTA tensor.")
    if not isinstance(model, PlaneTFIM):
        raise TypeError("model must be a PlaneTFIM.")
    if not isinstance(environment, PlaneEnvironmentOptions):
        raise TypeError(
            "environment must be a PlaneEnvironmentOptions instance."
        )
    environment = environment.validated()
    energy = _make_energy(shape, model, environment)
    compiled = jax.jit(jax.value_and_grad(energy, has_aux=True))
    expected_size = int(np.prod(shape))

    def evaluate(parameters):
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != (expected_size,):
            raise ValueError(
                f"parameters must have shape ({expected_size},)."
            )
        (value, observables), gradient = compiled(
            jnp.asarray(parameters)
        )
        value = float(value)
        observables = np.asarray(observables)
        gradient = np.asarray(gradient)
        if (
            not np.isfinite(value)
            or not np.all(np.isfinite(observables))
            or not np.all(np.isfinite(gradient))
        ):
            raise UnreliablePlaneEnvironmentError(
                "the automatic-differentiation boundary contraction "
                "became non-finite; increase boundary_bond_dim."
            )
        if np.any(np.abs(observables) > 1.0 + 1.0e-5):
            raise UnreliablePlaneEnvironmentError(
                "unphysical plane observable from the automatic-"
                "differentiation boundary environment; increase "
                "boundary_bond_dim."
            )
        return value, gradient

    return evaluate
