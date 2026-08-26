"""Spin Hamiltonians in the three-dimensional snake ordering."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ..operators import LatticeMPO

from .geometry import nearest_neighbor_bonds, validate_shape


def _ising_local_operators(basis, dtype):
    basis = str(basis).lower()
    if basis not in {"z", "x"}:
        raise ValueError("basis must be 'z' or 'x'.")
    x_operator = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
    z_operator = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=dtype)
    if basis == "z":
        return x_operator, z_operator
    return z_operator, x_operator


def transverse_field_ising_mpo(
    lattice_shape,
    *,
    coupling=1.0,
    field=1.0,
    ordering="continuous-snake",
    basis="z",
):
    r"""Build the open 3D TFIM MPO in snake order.

    The Hamiltonian is

    ``H = -coupling * sum_<ij> Z_i Z_j - field * sum_i X_i``.
    """

    shape = validate_shape(lattice_shape)
    nsites = int(np.prod(shape))
    bonds = nearest_neighbor_bonds(shape, ordering=ordering)
    max_distance = max((right - left for left, right in bonds), default=1)
    final = max_distance + 1
    mpo_bond_dim = final + 1
    dtype = np.result_type(coupling, field, float)
    identity = np.eye(2, dtype=dtype)
    field_operator, interaction_operator = _ising_local_operators(basis, dtype)
    starts = [[] for _ in range(nsites)]
    for left, right in bonds:
        starts[left].append(right - left)

    factors = []
    for site in range(nsites):
        factor = np.zeros((mpo_bond_dim, mpo_bond_dim, 2, 2), dtype=dtype)
        factor[0, 0] = identity
        factor[final, final] = identity
        factor[0, final] = -field * field_operator
        factor[1, final] = interaction_operator
        for remaining in range(2, max_distance + 1):
            factor[remaining, remaining - 1] = identity
        for distance in starts[site]:
            factor[0, distance] += -coupling * interaction_operator
        if site == 0:
            factor = factor[0:1]
        if site == nsites - 1:
            factor = factor[:, final : final + 1]
        factors.append(factor)
    return LatticeMPO(factors, lattice_shape=shape)


def transverse_field_ising_sparse(
    lattice_shape,
    *,
    coupling=1.0,
    field=1.0,
    max_sites=20,
    ordering="continuous-snake",
    basis="z",
):
    """Build a sparse reference Hamiltonian for small cubes only."""

    shape = validate_shape(lattice_shape)
    nsites = int(np.prod(shape))
    if nsites > max_sites:
        raise ValueError(
            "refusing to construct an exponentially large sparse Hamiltonian; "
            "use the MPO for this lattice."
        )
    dimension = 2**nsites
    bonds = nearest_neighbor_bonds(shape, ordering=ordering)
    rows = []
    columns = []
    values = []
    dimensions = (2,) * nsites
    basis = str(basis).lower()
    if basis not in {"z", "x"}:
        raise ValueError("basis must be 'z' or 'x'.")
    for column in range(dimension):
        configuration = np.unravel_index(column, dimensions)
        z_values = 1 - 2 * np.asarray(configuration)
        if basis == "z":
            diagonal = -float(coupling) * sum(
                z_values[left] * z_values[right] for left, right in bonds
            )
        else:
            diagonal = -float(field) * float(np.sum(z_values))
        rows.append(column)
        columns.append(column)
        values.append(diagonal)
        if basis == "z":
            for site in range(nsites):
                flipped = list(configuration)
                flipped[site] = 1 - flipped[site]
                row = np.ravel_multi_index(tuple(flipped), dimensions)
                rows.append(row)
                columns.append(column)
                values.append(-float(field))
        else:
            for left, right in bonds:
                flipped = list(configuration)
                flipped[left] = 1 - flipped[left]
                flipped[right] = 1 - flipped[right]
                row = np.ravel_multi_index(tuple(flipped), dimensions)
                rows.append(row)
                columns.append(column)
                values.append(-float(coupling))
    return sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(dimension, dimension),
        dtype=float,
    ).tocsr()
