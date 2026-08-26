"""Compare VUMPS with exact results for the transverse-field Ising chain.

Run from the repository root with

```
PYTHONPATH=. python -m pyqed._vumps.examples.tfim_comparison
```

The Hamiltonian convention is

    H = -J sum_n Z_n Z_{n+1} - g sum_n X_n.

VUMPS works directly in the thermodynamic limit. The two independent
references are the exact Jordan-Wigner integrals in that limit and sparse
exact diagonalization of a finite periodic ring.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from operator import index

import numpy as np
from scipy.integrate import quad
from scipy.sparse.linalg import LinearOperator, eigsh

from pyqed._vumps.operators import nearest_neighbor_energy, one_site_expectation
from pyqed._vumps.solver import VUMPSOptions, vumps


@dataclass(frozen=True)
class TFIMObservables:
    """Energy and local observables for one TFIM calculation."""

    method: str
    energy_density: float
    transverse_magnetization: float
    zz_correlation: float
    converged: bool | None = None
    iterations: int | None = None
    residual_norm: float | None = None


def _validate_parameters(coupling, field):
    coupling = float(coupling)
    field = float(field)
    if not np.isfinite(coupling) or coupling <= 0.0:
        raise ValueError("coupling must be finite and positive.")
    if not np.isfinite(field) or field < 0.0:
        raise ValueError("field must be finite and nonnegative.")
    return coupling, field


def _integer_parameter(value, name, *, minimum, maximum=None):
    try:
        value = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer.") from error
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be no larger than {maximum}.")
    return value


def tfim_bond_hamiltonian(*, coupling=1.0, field=1.5):
    """Return the two-site term whose infinite sum is the TFIM Hamiltonian."""

    coupling, field = _validate_parameters(coupling, field)
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    return -coupling * np.kron(z, z) - 0.5 * field * (
        np.kron(x, identity) + np.kron(identity, x)
    )


def _dispersion_root(momentum, coupling, field):
    value = (
        coupling * coupling
        + field * field
        - 2.0 * coupling * field * np.cos(momentum)
    )
    return np.sqrt(max(float(value), 0.0))


def exact_tfim_energy_density(*, coupling=1.0, field=1.5):
    """Return the exact infinite-chain ground-state energy per site."""

    coupling, field = _validate_parameters(coupling, field)
    integral, _error = quad(
        lambda momentum: _dispersion_root(momentum, coupling, field),
        0.0,
        2.0 * np.pi,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
        limit=300,
    )
    return -integral / (2.0 * np.pi)


def exact_tfim_transverse_magnetization(*, coupling=1.0, field=1.5):
    """Return the exact infinite-chain expectation value of ``X``."""

    coupling, field = _validate_parameters(coupling, field)

    def integrand(momentum):
        denominator = _dispersion_root(momentum, coupling, field)
        if denominator <= np.finfo(float).eps:
            return 0.0
        return (field - coupling * np.cos(momentum)) / denominator

    integral, _error = quad(
        integrand,
        0.0,
        2.0 * np.pi,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
        limit=300,
    )
    return integral / (2.0 * np.pi)


def exact_tfim_zz_correlation(*, coupling=1.0, field=1.5):
    """Return the exact infinite-chain nearest-neighbor ``ZZ`` expectation."""

    coupling, field = _validate_parameters(coupling, field)

    def integrand(momentum):
        denominator = _dispersion_root(momentum, coupling, field)
        if denominator <= np.finfo(float).eps:
            return 0.0
        return (coupling - field * np.cos(momentum)) / denominator

    # Pair k and pi-k before integration. This avoids subtracting two
    # order-one integrals when coupling is much smaller than the field.
    integral, _error = quad(
        lambda momentum: (
            integrand(momentum) + integrand(np.pi - momentum)
        ),
        0.0,
        0.5 * np.pi,
        epsabs=1.0e-14,
        epsrel=1.0e-12,
        limit=300,
    )
    return integral / np.pi


def finite_tfim_ground_state(*, num_sites=12, coupling=1.0, field=1.5):
    """Solve a finite periodic TFIM ring by sparse exact diagonalization."""

    coupling, field = _validate_parameters(coupling, field)
    num_sites = _integer_parameter(
        num_sites,
        "num_sites",
        minimum=2,
        maximum=20,
    )

    dimension = 1 << num_sites
    basis = np.arange(dimension, dtype=np.int64)
    diagonal = np.zeros(dimension)

    for site in range(num_sites):
        neighbor = (site + 1) % num_sites
        z_site = 1.0 - 2.0 * ((basis >> site) & 1)
        z_neighbor = 1.0 - 2.0 * ((basis >> neighbor) & 1)
        diagonal -= coupling * z_site * z_neighbor

    def apply_hamiltonian(vector):
        vector = np.asarray(vector)
        output = diagonal * vector
        for site in range(num_sites):
            output = output - field * vector[basis ^ (1 << site)]
        return output

    hamiltonian = LinearOperator(
        shape=(dimension, dimension),
        matvec=apply_hamiltonian,
        dtype=float,
    )
    initial = np.full(dimension, 1.0 / np.sqrt(dimension))
    eigenvalues, eigenvectors = eigsh(
        hamiltonian,
        k=1,
        which="SA",
        v0=initial,
        tol=1.0e-12,
    )
    ground_state = eigenvectors[:, 0]

    transverse_magnetization = 0.0j
    zz_correlation = 0.0
    probabilities = np.abs(ground_state) ** 2
    for site in range(num_sites):
        neighbor = (site + 1) % num_sites
        transverse_magnetization += np.vdot(
            ground_state,
            ground_state[basis ^ (1 << site)],
        )
        z_site = 1.0 - 2.0 * ((basis >> site) & 1)
        z_neighbor = 1.0 - 2.0 * ((basis >> neighbor) & 1)
        zz_correlation += float(np.dot(probabilities, z_site * z_neighbor))

    return TFIMObservables(
        method=f"periodic ED, L={num_sites}",
        energy_density=float(np.real(eigenvalues[0])) / num_sites,
        transverse_magnetization=float(
            np.real(transverse_magnetization)
        ) / num_sites,
        zz_correlation=zz_correlation / num_sites,
    )


def vumps_tfim_ground_state(
    *,
    bond_dim,
    coupling=1.0,
    field=1.5,
    seed=3,
    tolerance=1.0e-10,
    max_iterations=200,
):
    """Solve the infinite TFIM with one-site VUMPS."""

    bond_dim = _integer_parameter(bond_dim, "bond_dim", minimum=1)
    hamiltonian = tfim_bond_hamiltonian(coupling=coupling, field=field)
    result = vumps(
        hamiltonian,
        bond_dim=bond_dim,
        seed=seed,
        options=VUMPSOptions(
            max_iterations=max_iterations,
            tolerance=tolerance,
        ),
    )
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    return TFIMObservables(
        method=f"VUMPS, D={bond_dim}",
        energy_density=result.energy,
        transverse_magnetization=float(one_site_expectation(result.state, x)),
        zz_correlation=nearest_neighbor_energy(result.state, np.kron(z, z)),
        converged=result.converged,
        iterations=result.iterations,
        residual_norm=result.residual_norm,
    )


def compare_tfim_methods(
    *,
    bond_dimensions=(1, 2, 4),
    num_sites=12,
    coupling=1.0,
    field=1.5,
    seed=3,
    tolerance=1.0e-10,
    max_iterations=200,
):
    """Run exact thermodynamic, finite ED, and VUMPS calculations."""

    exact_energy = exact_tfim_energy_density(coupling=coupling, field=field)
    exact_magnetization = exact_tfim_transverse_magnetization(
        coupling=coupling,
        field=field,
    )
    exact_zz = exact_tfim_zz_correlation(coupling=coupling, field=field)
    results = [
        TFIMObservables(
            method="exact, infinite",
            energy_density=exact_energy,
            transverse_magnetization=exact_magnetization,
            zz_correlation=exact_zz,
        ),
        finite_tfim_ground_state(
            num_sites=num_sites,
            coupling=coupling,
            field=field,
        ),
    ]
    for bond_dim in bond_dimensions:
        results.append(
            vumps_tfim_ground_state(
                bond_dim=bond_dim,
                coupling=coupling,
                field=field,
                seed=seed,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
        )
    return tuple(results)


def _print_results(results, *, coupling, field):
    reference = results[0]
    print("Transverse-field Ising chain")
    print(f"H = -{coupling:g} sum(Z_n Z_(n+1)) - {field:g} sum(X_n)")
    print()
    print(
        f"{'method':<20} {'energy/site':>15} {'|error|':>11} "
        f"{'<X>':>13} {'|error|':>11} {'<ZZ>':>13} {'|error|':>11}"
    )
    print("-" * 101)
    for result in results:
        print(
            f"{result.method:<20} "
            f"{result.energy_density:15.10f} "
            f"{abs(result.energy_density - reference.energy_density):11.3e} "
            f"{result.transverse_magnetization:13.10f} "
            f"{abs(result.transverse_magnetization - reference.transverse_magnetization):11.3e} "
            f"{result.zz_correlation:13.10f} "
            f"{abs(result.zz_correlation - reference.zz_correlation):11.3e}"
        )
    print()
    for result in results:
        if result.converged is not None:
            print(
                f"{result.method}: converged={result.converged}, "
                f"iterations={result.iterations}, "
                f"residual={result.residual_norm:.3e}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--field", type=float, default=1.5)
    parser.add_argument("--sites", type=int, default=12)
    parser.add_argument(
        "--bond-dimensions",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8),
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--max-iterations", type=int, default=200)
    arguments = parser.parse_args(argv)

    results = compare_tfim_methods(
        bond_dimensions=arguments.bond_dimensions,
        num_sites=arguments.sites,
        coupling=arguments.coupling,
        field=arguments.field,
        seed=arguments.seed,
        tolerance=arguments.tolerance,
        max_iterations=arguments.max_iterations,
    )
    _print_results(
        results,
        coupling=arguments.coupling,
        field=arguments.field,
    )


if __name__ == "__main__":
    main()
