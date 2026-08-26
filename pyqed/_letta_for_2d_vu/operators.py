"""Nearest-neighbor operators for uniform infinite cylinders."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from pyqed._vuletta.operators import (
    energy_density,
    one_site_expectation,
    two_site_expectation,
)

from .state import UniformCylinderLETTA, _positive_integer, _transverse_boundary


def _kron_all(operators):
    result = np.asarray(operators[0])
    for operator in operators[1:]:
        result = np.kron(result, operator)
    return result


@lru_cache(maxsize=32)
def _spin_half_column_operators(width):
    width = _positive_integer(width, "width")
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    identities = (identity,) * width
    x_operators = []
    z_operators = []
    for row in range(width):
        factors = list(identities)
        factors[row] = x
        x_operators.append(_kron_all(factors))
        factors = list(identities)
        factors[row] = z
        z_operators.append(_kron_all(factors))
    return identity, tuple(x_operators), tuple(z_operators)


def _transverse_edges(width, boundary):
    if boundary == "open":
        return tuple((row, row + 1) for row in range(width - 1))
    if width == 1:
        return ()
    return tuple((row, (row + 1) % width) for row in range(width))


@dataclass(frozen=True)
class CylinderTFIM:
    """Two-column density for a width-by-infinity TFIM."""

    width: int
    coupling: float
    field: float
    transverse_boundary: str
    local_density: np.ndarray
    x_average: np.ndarray
    horizontal_zz_average: np.ndarray
    transverse_zz_average: np.ndarray
    transverse_bond_count: int

    @property
    def local_physical_dim(self):
        return 2

    @property
    def column_dim(self):
        return 2**self.width


def tfim_cylinder_hamiltonian(
    width,
    *,
    coupling=1.0,
    field=1.0,
    transverse_boundary="periodic",
    max_column_dim=32,
):
    """Build the exact two-column TFIM density for an infinite cylinder."""

    width = _positive_integer(width, "width")
    boundary = _transverse_boundary(transverse_boundary)
    coupling = float(coupling)
    field = float(field)
    if not np.isfinite(coupling) or not np.isfinite(field):
        raise ValueError("coupling and field must be finite.")
    column_dim = 2**width
    if max_column_dim is not None and column_dim > int(max_column_dim):
        raise ValueError(
            "the dense column operator exceeds max_column_dim; "
            "reduce width or raise the explicit limit."
        )

    identity_local, x_operators, z_operators = (
        _spin_half_column_operators(width)
    )
    identity_column = np.eye(column_dim)
    x_sum = np.add.reduce(x_operators)
    transverse_edges = _transverse_edges(width, boundary)
    if transverse_edges:
        transverse_zz_sum = np.add.reduce(
            tuple(
                z_operators[left] @ z_operators[right]
                for left, right in transverse_edges
            )
        )
    else:
        transverse_zz_sum = np.zeros(
            (column_dim, column_dim),
            dtype=float,
        )
    column_hamiltonian = (
        -coupling * transverse_zz_sum
        - field * x_sum
    )
    horizontal_zz_sum = np.add.reduce(
        tuple(
            np.kron(z_operator, z_operator)
            for z_operator in z_operators
        )
    )
    local_density = (
        -coupling * horizontal_zz_sum
        + 0.5 * np.kron(column_hamiltonian, identity_column)
        + 0.5 * np.kron(identity_column, column_hamiltonian)
    )
    horizontal_zz_average = horizontal_zz_sum / width
    transverse_zz_average = (
        transverse_zz_sum / len(transverse_edges)
        if transverse_edges
        else transverse_zz_sum
    )
    return CylinderTFIM(
        width=width,
        coupling=coupling,
        field=field,
        transverse_boundary=boundary,
        local_density=local_density.reshape(
            column_dim,
            column_dim,
            column_dim,
            column_dim,
        ),
        x_average=x_sum / width,
        horizontal_zz_average=horizontal_zz_average.reshape(
            column_dim,
            column_dim,
            column_dim,
            column_dim,
        ),
        transverse_zz_average=transverse_zz_average,
        transverse_bond_count=len(transverse_edges),
    )


def _validate_state_model(state, model):
    if not isinstance(state, UniformCylinderLETTA):
        raise TypeError("state must be a UniformCylinderLETTA.")
    if not isinstance(model, CylinderTFIM):
        raise TypeError("model must be a CylinderTFIM.")
    if state.width != model.width:
        raise ValueError("state and model widths do not agree.")
    if state.local_physical_dim != model.local_physical_dim:
        raise ValueError("state and model local dimensions do not agree.")
    if state.transverse_boundary != model.transverse_boundary:
        raise ValueError("state and model transverse boundaries do not agree.")


def cylinder_energy_density(state, model):
    """Return the TFIM energy per microscopic spin."""

    _validate_state_model(state, model)
    return energy_density(state.uniform_state, model.local_density) / model.width


def transverse_magnetization(state):
    """Return the column-averaged expectation value of X."""

    if not isinstance(state, UniformCylinderLETTA):
        raise TypeError("state must be a UniformCylinderLETTA.")
    if state.local_physical_dim != 2:
        raise ValueError("spin observables require local_physical_dim=2.")
    _identity, x_operators, _z_operators = _spin_half_column_operators(
        state.width
    )
    return one_site_expectation(
        state.uniform_state,
        np.add.reduce(x_operators) / state.width,
    )


def horizontal_zz_expectation(state):
    """Return ZZ averaged over bonds parallel to the infinite direction."""

    if not isinstance(state, UniformCylinderLETTA):
        raise TypeError("state must be a UniformCylinderLETTA.")
    _identity, _x_operators, z_operators = _spin_half_column_operators(
        state.width
    )
    operator = np.add.reduce(
        tuple(np.kron(z_operator, z_operator) for z_operator in z_operators)
    ) / state.width
    return two_site_expectation(state.uniform_state, operator)


def transverse_zz_expectation(state):
    """Return ZZ averaged over transverse nearest-neighbor bonds."""

    if not isinstance(state, UniformCylinderLETTA):
        raise TypeError("state must be a UniformCylinderLETTA.")
    _identity, _x_operators, z_operators = _spin_half_column_operators(
        state.width
    )
    edges = _transverse_edges(state.width, state.transverse_boundary)
    if not edges:
        return 0.0
    operator = np.add.reduce(
        tuple(
            z_operators[left] @ z_operators[right]
            for left, right in edges
        )
    ) / len(edges)
    return one_site_expectation(state.uniform_state, operator)
