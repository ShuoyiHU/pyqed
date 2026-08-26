"""Dimension-independent MPO and observable helpers for finite LETTA."""

from __future__ import annotations

import numpy as np
import opt_einsum as oe
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .state import LatticeLETTA, _validate_lattice_shape


class LatticeMPO:
    """Open-boundary matrix product operator in bra-ket convention."""

    def __init__(self, factors, *, lattice_shape=None):
        factors = tuple(np.asarray(factor).copy() for factor in factors)
        if not factors:
            raise ValueError("an MPO must contain at least one factor.")
        physical_dim = factors[0].shape[2] if factors[0].ndim == 4 else None
        for site, factor in enumerate(factors):
            if factor.ndim != 4:
                raise ValueError("MPO factors must have four axes.")
            if factor.shape[2:] != (physical_dim, physical_dim):
                raise ValueError("MPO physical dimensions must be uniform and square.")
            if site == 0 and factor.shape[0] != 1:
                raise ValueError("the first MPO left bond must have dimension one.")
            if site == len(factors) - 1 and factor.shape[1] != 1:
                raise ValueError("the last MPO right bond must have dimension one.")
            if site and factors[site - 1].shape[1] != factor.shape[0]:
                raise ValueError(f"MPO bond mismatch before site {site}.")
            if not np.all(np.isfinite(factor)):
                raise ValueError("MPO factors must contain finite values.")
        if lattice_shape is not None:
            lattice_shape = _validate_lattice_shape(lattice_shape)
            if int(np.prod(lattice_shape)) != len(factors):
                raise ValueError("MPO length does not match lattice_shape.")
        self.factors = factors
        self.transitions = tuple(
            tuple(
                (left, right, factor[left, right])
                for left in range(factor.shape[0])
                for right in range(factor.shape[1])
                if np.any(factor[left, right] != 0)
            )
            for factor in factors
        )
        self.lattice_shape = lattice_shape
        self.physical_dim = int(physical_dim)

    @property
    def nsites(self):
        return len(self.factors)

    @property
    def bond_dimensions(self):
        return tuple(factor.shape[1] for factor in self.factors[:-1])

    @property
    def shape(self):
        dimension = self.physical_dim**self.nsites
        return dimension, dimension

    def to_dense(self, *, max_sites=12):
        """Materialize the MPO for tests and small exact references only."""

        if self.nsites > max_sites:
            raise ValueError(
                "refusing to materialize a dense MPO with more than "
                f"{max_sites} sites."
            )
        virtual = list(range(self.nsites + 1))
        bra = list(range(self.nsites + 1, 2 * self.nsites + 1))
        ket = list(range(2 * self.nsites + 1, 3 * self.nsites + 1))
        arguments = []
        for site, factor in enumerate(self.factors):
            arguments.extend(
                [factor, [virtual[site], virtual[site + 1], bra[site], ket[site]]]
            )
        tensor = oe.contract(*arguments, bra + ket, optimize="greedy")
        return tensor.reshape(self.shape)


def identity_mpo(nsites, physical_dim=2, *, lattice_shape=None):
    """Return an identity MPO with virtual bond dimension one."""

    nsites = int(nsites)
    physical_dim = int(physical_dim)
    if nsites <= 0 or physical_dim <= 0:
        raise ValueError("nsites and physical_dim must be positive.")
    identity = np.eye(physical_dim).reshape(1, 1, physical_dim, physical_dim)
    return LatticeMPO(
        [identity.copy() for _ in range(nsites)],
        lattice_shape=lattice_shape,
    )


def exact_ground_state(hamiltonian):
    """Return the lowest eigenvalue and normalized eigenvector."""

    dimension = hamiltonian.shape[0]
    if hamiltonian.shape != (dimension, dimension):
        raise ValueError("hamiltonian must be square.")
    if dimension <= 256:
        dense = (
            hamiltonian.toarray()
            if sparse.issparse(hamiltonian)
            else np.asarray(hamiltonian)
        )
        values, vectors = np.linalg.eigh(dense)
        return float(np.real(values[0])), vectors[:, 0]
    values, vectors = eigsh(hamiltonian, k=1, which="SA", tol=1.0e-12)
    return float(np.real(values[0])), vectors[:, 0]


def _apply_one_site(vector, operator, site, physical_dim, nsites):
    operator = np.asarray(operator)
    if operator.shape != (physical_dim, physical_dim):
        raise ValueError("operator has incompatible local dimensions.")
    tensor = np.asarray(vector).reshape((physical_dim,) * nsites)
    applied = np.tensordot(operator, tensor, axes=([1], [site]))
    applied = np.moveaxis(applied, 0, site)
    return applied.reshape(-1)


def one_site_expectation(state, operator, site):
    if not isinstance(state, LatticeLETTA):
        raise TypeError("state must be a LatticeLETTA.")
    return state_vector_one_site_expectation(
        state.state_vector(),
        operator,
        site,
        physical_dim=state.physical_dim,
    )


def _infer_nsites(vector, physical_dim):
    vector = np.asarray(vector).reshape(-1)
    nsites = int(round(np.log(vector.size) / np.log(physical_dim)))
    if physical_dim**nsites != vector.size:
        raise ValueError("vector size is not a power of physical_dim.")
    return vector, nsites


def state_vector_one_site_expectation(vector, operator, site, *, physical_dim=2):
    vector, nsites = _infer_nsites(vector, physical_dim)
    applied = _apply_one_site(
        vector,
        operator,
        int(site),
        physical_dim,
        nsites,
    )
    return np.real_if_close(np.vdot(vector, applied) / np.vdot(vector, vector)).item()


def two_site_expectation(state, operator_a, site_a, operator_b, site_b):
    if not isinstance(state, LatticeLETTA):
        raise TypeError("state must be a LatticeLETTA.")
    return state_vector_two_site_expectation(
        state.state_vector(),
        operator_a,
        site_a,
        operator_b,
        site_b,
        physical_dim=state.physical_dim,
    )


def state_vector_two_site_expectation(
    vector,
    operator_a,
    site_a,
    operator_b,
    site_b,
    *,
    physical_dim=2,
):
    vector, nsites = _infer_nsites(vector, physical_dim)
    applied = _apply_one_site(
        vector,
        operator_b,
        int(site_b),
        physical_dim,
        nsites,
    )
    applied = _apply_one_site(
        applied,
        operator_a,
        int(site_a),
        physical_dim,
        nsites,
    )
    return np.real_if_close(np.vdot(vector, applied) / np.vdot(vector, vector)).item()
