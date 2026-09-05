"""Small exact MPOs for the LETTA condensed-matter benchmarks.

All lattices use open boundaries and NumPy C-order site numbering.  A 1D
chain is represented by the degenerate LETTA shape ``(1, length)``; a 2D
case uses the requested ``(rows, columns)`` rectangle.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..operators import LatticeMPO


MODEL_NAMES = ("ising", "heisenberg", "bose_hubbard", "fermi_hubbard")
MODEL_CASES = tuple(
    (name, dimension) for dimension in ("1d", "2d") for name in MODEL_NAMES
)


@dataclass(frozen=True)
class ProductTerm:
    """One coefficient times a product of local site operators."""

    coefficient: complex
    operators: Mapping[int, np.ndarray]


@dataclass(frozen=True)
class CondensedModel:
    """A fully specified benchmark Hamiltonian."""

    name: str
    dimension: str
    lattice_shape: tuple[int, int]
    physical_dim: int
    parameters: Mapping[str, float | int]
    bonds: tuple[tuple[int, int], ...]
    terms: tuple[ProductTerm, ...]
    mpo: LatticeMPO

    @property
    def nsites(self):
        return int(np.prod(self.lattice_shape))

    @property
    def hilbert_dim(self):
        return self.physical_dim**self.nsites


def _positive_integer(value, name):
    try:
        value = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer.") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def nearest_neighbor_bonds(dimension, size):
    """Return the LETTA shape and ordered open-boundary nearest-neighbor bonds."""

    dimension = str(dimension).lower()
    if dimension == "1d":
        length = _positive_integer(size, "length")
        if length < 2:
            raise ValueError("a benchmark chain needs at least two sites.")
        shape = (1, length)
    elif dimension == "2d":
        try:
            rows, columns = size
        except (TypeError, ValueError) as error:
            raise ValueError("2D size must be a (rows, columns) pair.") from error
        rows = _positive_integer(rows, "rows")
        columns = _positive_integer(columns, "columns")
        if rows < 2 or columns < 2:
            raise ValueError("a 2D benchmark needs at least two rows and columns.")
        shape = (rows, columns)
    else:
        raise ValueError("dimension must be '1d' or '2d'.")

    bonds = []
    rows, columns = shape
    for row in range(rows):
        for column in range(columns):
            site = row * columns + column
            if column + 1 < columns:
                bonds.append((site, site + 1))
            if row + 1 < rows:
                bonds.append((site, site + columns))
    return shape, tuple(bonds)


def _validated_term(term, nsites, physical_dim):
    if not isinstance(term, ProductTerm):
        raise TypeError("terms must be ProductTerm instances.")
    coefficient = complex(term.coefficient)
    if not np.isfinite(coefficient):
        raise ValueError("product-term coefficients must be finite.")
    operators = {}
    for site, operator in term.operators.items():
        site = index(site)
        if site < 0 or site >= nsites:
            raise ValueError("a product-term site is outside the lattice.")
        operator = np.asarray(operator)
        if operator.shape != (physical_dim, physical_dim):
            raise ValueError("a product-term operator has incompatible dimensions.")
        if not np.all(np.isfinite(operator)):
            raise ValueError("product-term operators must be finite.")
        operators[site] = operator
    return coefficient, operators


def product_terms_to_mpo(
    terms,
    *,
    nsites,
    physical_dim,
    lattice_shape,
):
    """Build an exact sparse-channel MPO from a sum of product terms.

    Each product term receives one independent virtual channel.  This is not
    a minimal-bond MPO, but it preserves the exact Hamiltonian and leaves only
    ``O(number_of_terms)`` nonzero transitions for the LETTA sparse-MPO path.
    """

    nsites = _positive_integer(nsites, "nsites")
    physical_dim = _positive_integer(physical_dim, "physical_dim")
    terms = tuple(terms)
    if not terms:
        raise ValueError("at least one product term is required.")
    validated = tuple(
        _validated_term(term, nsites, physical_dim) for term in terms
    )
    dtype = np.result_type(
        complex,
        *(operator.dtype for _, operators in validated for operator in operators.values()),
    )
    identity = np.eye(physical_dim, dtype=dtype)

    if nsites == 1:
        factor = np.zeros((1, 1, physical_dim, physical_dim), dtype=dtype)
        for coefficient, operators in validated:
            factor[0, 0] += coefficient * operators.get(0, identity)
        return LatticeMPO((np.real_if_close(factor),), lattice_shape=lattice_shape)

    width = len(validated)
    factors = []
    first = np.zeros((1, width, physical_dim, physical_dim), dtype=dtype)
    for channel, (coefficient, operators) in enumerate(validated):
        first[0, channel] = coefficient * operators.get(0, identity)
    factors.append(first)
    for site in range(1, nsites - 1):
        factor = np.zeros((width, width, physical_dim, physical_dim), dtype=dtype)
        for channel, (_, operators) in enumerate(validated):
            factor[channel, channel] = operators.get(site, identity)
        factors.append(factor)
    last = np.zeros((width, 1, physical_dim, physical_dim), dtype=dtype)
    for channel, (_, operators) in enumerate(validated):
        last[channel, 0] = operators.get(nsites - 1, identity)
    factors.append(last)
    return LatticeMPO(
        tuple(np.real_if_close(factor) for factor in factors),
        lattice_shape=lattice_shape,
    )


def _number(value, name):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _parameters(defaults, supplied):
    extra = set(supplied) - set(defaults)
    if extra:
        names = ", ".join(sorted(extra))
        raise ValueError(f"unknown model parameter(s): {names}.")
    result = dict(defaults)
    result.update(supplied)
    for name in result:
        if name != "max_occupancy":
            result[name] = _number(result[name], name)
    return result


def _ising_terms(nsites, bonds, parameters):
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    terms = [
        ProductTerm(-parameters["J"], {left: z, right: z})
        for left, right in bonds
    ]
    terms.extend(
        ProductTerm(-parameters["h"], {site: x}) for site in range(nsites)
    )
    return 2, tuple(terms)


def _heisenberg_terms(nsites, bonds, parameters):
    sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    sm = sp.T
    sz = 0.5 * np.diag([1.0, -1.0])
    terms = []
    for left, right in bonds:
        terms.extend(
            (
                ProductTerm(0.5 * parameters["J"], {left: sp, right: sm}),
                ProductTerm(0.5 * parameters["J"], {left: sm, right: sp}),
                ProductTerm(
                    parameters["J"] * parameters["delta"],
                    {left: sz, right: sz},
                ),
            )
        )
    terms.extend(
        ProductTerm(-parameters["h"], {site: sz}) for site in range(nsites)
    )
    return 2, tuple(terms)


def _bose_hubbard_terms(nsites, bonds, parameters):
    maximum = _positive_integer(parameters["max_occupancy"], "max_occupancy")
    parameters["max_occupancy"] = maximum
    physical_dim = maximum + 1
    annihilation = np.diag(np.sqrt(np.arange(1, physical_dim)), 1)
    creation = annihilation.T
    number = np.diag(np.arange(physical_dim, dtype=float))
    onsite = (
        0.5 * parameters["U"] * number @ (number - np.eye(physical_dim))
        - parameters["mu"] * number
    )
    terms = []
    for left, right in bonds:
        terms.extend(
            (
                ProductTerm(
                    -parameters["t"], {left: creation, right: annihilation}
                ),
                ProductTerm(
                    -parameters["t"], {left: annihilation, right: creation}
                ),
            )
        )
    terms.extend(ProductTerm(1.0, {site: onsite}) for site in range(nsites))
    return physical_dim, tuple(terms)


def _fermi_hubbard_terms(nsites, bonds, parameters):
    cup = np.zeros((4, 4))
    cup[0, 1] = cup[2, 3] = 1.0
    cdown = np.zeros((4, 4))
    cdown[0, 2] = 1.0
    cdown[1, 3] = -1.0
    parity = np.diag([1.0, -1.0, -1.0, 1.0])
    nup = cup.T @ cup
    ndown = cdown.T @ cdown
    onsite = (
        parameters["U"] * nup @ ndown
        - parameters["mu"] * (nup + ndown)
    )
    terms = []
    for left, right in bonds:
        if left >= right:
            raise ValueError("fermionic bonds must follow the MPS site ordering.")
        string = {site: parity for site in range(left + 1, right)}
        for annihilation in (cup, cdown):
            forward = dict(string)
            forward[left] = annihilation.T @ parity
            forward[right] = annihilation
            backward = dict(string)
            backward[left] = parity @ annihilation
            backward[right] = annihilation.T
            terms.extend(
                (
                    ProductTerm(-parameters["t"], forward),
                    ProductTerm(-parameters["t"], backward),
                )
            )
    terms.extend(ProductTerm(1.0, {site: onsite}) for site in range(nsites))
    return 4, tuple(terms)


def build_model(name, dimension, size, **supplied_parameters):
    """Build one of the four model families in one or two dimensions."""

    aliases = {
        "tfim": "ising",
        "xxz": "heisenberg",
        "bose": "bose_hubbard",
        "fermi": "fermi_hubbard",
        "hubbard": "fermi_hubbard",
    }
    name = aliases.get(str(name).lower(), str(name).lower())
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}.")
    dimension = str(dimension).lower()
    lattice_shape, bonds = nearest_neighbor_bonds(dimension, size)
    nsites = int(np.prod(lattice_shape))

    if name == "ising":
        parameters = _parameters({"J": 1.0, "h": 1.0}, supplied_parameters)
        physical_dim, terms = _ising_terms(nsites, bonds, parameters)
    elif name == "heisenberg":
        parameters = _parameters(
            {"J": 1.0, "delta": 1.0, "h": 0.0}, supplied_parameters
        )
        physical_dim, terms = _heisenberg_terms(nsites, bonds, parameters)
    elif name == "bose_hubbard":
        parameters = _parameters(
            {"t": 1.0, "U": 4.0, "mu": 2.0, "max_occupancy": 2},
            supplied_parameters,
        )
        physical_dim, terms = _bose_hubbard_terms(nsites, bonds, parameters)
    else:
        parameters = _parameters(
            {"t": 1.0, "U": 4.0, "mu": 2.0}, supplied_parameters
        )
        physical_dim, terms = _fermi_hubbard_terms(nsites, bonds, parameters)

    mpo = product_terms_to_mpo(
        terms,
        nsites=nsites,
        physical_dim=physical_dim,
        lattice_shape=lattice_shape,
    )
    return CondensedModel(
        name=name,
        dimension=dimension,
        lattice_shape=lattice_shape,
        physical_dim=physical_dim,
        parameters=MappingProxyType(dict(parameters)),
        bonds=bonds,
        terms=terms,
        mpo=mpo,
    )


__all__ = [
    "CondensedModel",
    "MODEL_CASES",
    "MODEL_NAMES",
    "ProductTerm",
    "build_model",
    "nearest_neighbor_bonds",
    "product_terms_to_mpo",
]
