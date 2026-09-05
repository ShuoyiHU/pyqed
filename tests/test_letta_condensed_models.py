"""Exact contracts for the condensed-model LETTA benchmarks."""

from __future__ import annotations

import numpy as np
import pytest

from pyqed._letta_one_site_opt.benchmarks.condensed_models import (
    MODEL_CASES,
    ProductTerm,
    build_model,
    nearest_neighbor_bonds,
    product_terms_to_mpo,
)


def _kron_product(operators):
    result = np.asarray(operators[0])
    for operator in operators[1:]:
        result = np.kron(result, operator)
    return result


def _embed(operator, site, nsites, physical_dim):
    identity = np.eye(physical_dim)
    return _kron_product(
        [operator if cursor == site else identity for cursor in range(nsites)]
    )


def _spin_dense(model):
    nsites = model.nsites
    identity = np.eye(2)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sz = np.diag([1.0, -1.0])
    sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    sm = sp.T
    hamiltonian = np.zeros((2**nsites, 2**nsites))
    if model.name == "ising":
        for left, right in model.bonds:
            operators = [identity] * nsites
            operators[left] = sz
            operators[right] = sz
            hamiltonian -= model.parameters["J"] * _kron_product(operators)
        for site in range(nsites):
            hamiltonian -= model.parameters["h"] * _embed(sx, site, nsites, 2)
        return hamiltonian
    for left, right in model.bonds:
        for coefficient, op_left, op_right in (
            (0.5, sp, sm),
            (0.5, sm, sp),
            (model.parameters["delta"], 0.5 * sz, 0.5 * sz),
        ):
            operators = [identity] * nsites
            operators[left] = op_left
            operators[right] = op_right
            hamiltonian += (
                model.parameters["J"] * coefficient * _kron_product(operators)
            )
    for site in range(nsites):
        hamiltonian -= model.parameters["h"] * _embed(0.5 * sz, site, nsites, 2)
    return hamiltonian


def _bose_dense(model):
    d = model.physical_dim
    nsites = model.nsites
    annihilation = np.diag(np.sqrt(np.arange(1, d)), 1)
    creation = annihilation.T
    number = np.diag(np.arange(d, dtype=float))
    hamiltonian = np.zeros((d**nsites, d**nsites))
    for left, right in model.bonds:
        for op_left, op_right in (
            (creation, annihilation),
            (annihilation, creation),
        ):
            operators = [np.eye(d)] * nsites
            operators[left] = op_left
            operators[right] = op_right
            hamiltonian -= model.parameters["t"] * _kron_product(operators)
    onsite = (
        0.5 * model.parameters["U"] * number @ (number - np.eye(d))
        - model.parameters["mu"] * number
    )
    for site in range(nsites):
        hamiltonian += _embed(onsite, site, nsites, d)
    return hamiltonian


def _fermi_dense(model):
    nsites = model.nsites
    identity = np.eye(4)
    parity = np.diag([1.0, -1.0, -1.0, 1.0])
    cup = np.zeros((4, 4))
    cup[0, 1] = cup[2, 3] = 1.0
    cdown = np.zeros((4, 4))
    cdown[0, 2] = 1.0
    cdown[1, 3] = -1.0

    def global_annihilation(site, local):
        return _kron_product(
            [
                parity
                if cursor < site
                else local
                if cursor == site
                else identity
                for cursor in range(nsites)
            ]
        )

    annihilators = {
        (site, spin): global_annihilation(site, local)
        for site in range(nsites)
        for spin, local in (("up", cup), ("down", cdown))
    }
    dimension = 4**nsites
    hamiltonian = np.zeros((dimension, dimension))
    for left, right in model.bonds:
        for spin in ("up", "down"):
            ci = annihilators[left, spin]
            cj = annihilators[right, spin]
            hamiltonian -= model.parameters["t"] * (
                ci.T @ cj + cj.T @ ci
            )
    nup = cup.T @ cup
    ndown = cdown.T @ cdown
    onsite = (
        model.parameters["U"] * nup @ ndown
        - model.parameters["mu"] * (nup + ndown)
    )
    for site in range(nsites):
        hamiltonian += _embed(onsite, site, nsites, 4)
    return hamiltonian


@pytest.mark.parametrize("dimension", ["1d", "2d"])
@pytest.mark.parametrize(
    "name", ["ising", "heisenberg", "bose_hubbard", "fermi_hubbard"]
)
def test_registry_contains_four_models_in_each_dimension(name, dimension):
    assert (name, dimension) in MODEL_CASES


@pytest.mark.parametrize(
    ("dimension", "size", "expected_shape", "expected_bonds"),
    [
        ("1d", 4, (1, 4), ((0, 1), (1, 2), (2, 3))),
        (
            "2d",
            (2, 3),
            (2, 3),
            ((0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5)),
        ),
    ],
)
def test_open_nearest_neighbor_graph(
    dimension, size, expected_shape, expected_bonds
):
    shape, bonds = nearest_neighbor_bonds(dimension, size)
    assert shape == expected_shape
    assert bonds == expected_bonds


@pytest.mark.parametrize(
    ("name", "parameters", "reference"),
    [
        ("ising", {"J": 0.7, "h": 1.2}, _spin_dense),
        ("heisenberg", {"J": 0.8, "delta": 1.3, "h": -0.2}, _spin_dense),
        (
            "bose_hubbard",
            {"t": 0.4, "U": 1.7, "mu": 0.3, "max_occupancy": 2},
            _bose_dense,
        ),
        ("fermi_hubbard", {"t": 0.6, "U": 2.1, "mu": 0.9}, _fermi_dense),
    ],
)
@pytest.mark.parametrize(("dimension", "size"), [("1d", 3), ("2d", (2, 2))])
def test_mpo_matches_independent_dense_hamiltonian(
    name, parameters, reference, dimension, size
):
    model = build_model(name, dimension, size, **parameters)
    dense = model.mpo.to_dense(max_sites=model.nsites)
    expected_dim = (
        3
        if name == "bose_hubbard"
        else 4
        if name == "fermi_hubbard"
        else 2
    )
    assert model.physical_dim == expected_dim
    assert np.allclose(dense, dense.T.conj(), atol=1.0e-12)
    assert np.allclose(dense, reference(model), atol=1.0e-12)


def test_product_term_mpo_combines_single_site_terms():
    z = np.diag([1.0, -1.0])
    mpo = product_terms_to_mpo(
        [ProductTerm(2.0, {0: z}), ProductTerm(-0.5, {0: np.eye(2)})],
        nsites=1,
        physical_dim=2,
        lattice_shape=(1, 1),
    )
    assert np.allclose(
        mpo.to_dense(max_sites=1), 2.0 * z - 0.5 * np.eye(2)
    )


@pytest.mark.parametrize(
    ("name", "dimension", "size", "parameters"),
    [
        ("unknown", "1d", 2, {}),
        ("ising", "3d", 2, {}),
        ("ising", "1d", 1, {}),
        ("bose_hubbard", "1d", 2, {"max_occupancy": 0}),
    ],
)
def test_invalid_model_parameters_are_rejected(
    name, dimension, size, parameters
):
    with pytest.raises((TypeError, ValueError)):
        build_model(name, dimension, size, **parameters)
