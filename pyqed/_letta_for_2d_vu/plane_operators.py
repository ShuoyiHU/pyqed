"""Square-lattice transverse-field Ising observables for plane LETTA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .plane_environment import (
    PlaneEnvironmentOptions,
    contract_plane_window,
    contraction_ratio,
    double_layer_cell,
)
from .plane_state import UniformPlaneLETTA


class UnreliablePlaneEnvironmentError(ValueError):
    """Raised when boundary truncation produces an unphysical expectation."""


@dataclass(frozen=True)
class PlaneTFIM:
    coupling: float = 1.0
    field: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.coupling) or not np.isfinite(self.field):
            raise ValueError("coupling and field must be finite.")


@dataclass(frozen=True)
class PlaneObservableEstimate:
    window_size: int
    boundary_bond_dimension: int
    transverse_magnetization: float
    horizontal_zz: float
    vertical_zz: float
    maximum_boundary_bond_dimension: int
    discarded_weight: float


@dataclass(frozen=True)
class PlaneObservables:
    transverse_magnetization: float
    horizontal_zz: float
    vertical_zz: float
    window_size: int
    converged: bool
    window_converged: bool
    boundary_converged: bool
    window_change: float
    boundary_change: float
    estimates: tuple[PlaneObservableEstimate, ...]
    boundary_estimates: tuple[PlaneObservableEstimate, ...]
    maximum_boundary_bond_dimension: int
    discarded_weight: float


def tfim_square_lattice(*, coupling=1.0, field=1.0):
    """Return the infinite square-lattice TFIM parameters."""

    return PlaneTFIM(float(coupling), float(field))


def _spin_operators(local_dim):
    if local_dim != 2:
        raise ValueError("TFIM observables require local_physical_dim=2.")
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    return identity, x, z


def _estimate_observables(
    state,
    size,
    environment,
    cells,
    *,
    boundary_bond_dim=None,
):
    identity_cell, x_cell, z_cell = cells
    center = size // 2
    if boundary_bond_dim is None:
        boundary_bond_dim = environment.boundary_bond_dim
    contraction_options = {
        "boundary_bond_dim": boundary_bond_dim,
        "cutoff": environment.cutoff,
    }
    norm = contract_plane_window(
        state,
        identity_cell,
        size,
        **contraction_options,
    )
    transverse = contract_plane_window(
        state,
        identity_cell,
        size,
        replacements={(center, center): x_cell},
        **contraction_options,
    )
    horizontal = contract_plane_window(
        state,
        identity_cell,
        size,
        replacements={
            (center, center): z_cell,
            (center, center + 1): z_cell,
        },
        **contraction_options,
    )
    vertical = contract_plane_window(
        state,
        identity_cell,
        size,
        replacements={
            (center, center): z_cell,
            (center + 1, center): z_cell,
        },
        **contraction_options,
    )
    contractions = (norm, transverse, horizontal, vertical)
    estimate = PlaneObservableEstimate(
        window_size=size,
        boundary_bond_dimension=boundary_bond_dim,
        transverse_magnetization=contraction_ratio(transverse, norm),
        horizontal_zz=contraction_ratio(horizontal, norm),
        vertical_zz=contraction_ratio(vertical, norm),
        maximum_boundary_bond_dimension=max(
            item.maximum_bond_dimension for item in contractions
        ),
        discarded_weight=max(item.discarded_weight for item in contractions),
    )
    physicality_tolerance = 1.0e-5
    values = (
        estimate.transverse_magnetization,
        estimate.horizontal_zz,
        estimate.vertical_zz,
    )
    if any(abs(value) > 1.0 + physicality_tolerance for value in values):
        raise UnreliablePlaneEnvironmentError(
            "unphysical plane observable from the truncated boundary "
            f"environment at window {size}: "
            f"<X>={values[0]:.8g}, <ZZ-x>={values[1]:.8g}, "
            f"<ZZ-y>={values[2]:.8g}. Increase boundary_bond_dim or "
            "reduce the optimization step."
        )
    return estimate


def _observable_change(left, right):
    return max(
        abs(
            right.transverse_magnetization
            - left.transverse_magnetization
        ),
        abs(right.horizontal_zz - left.horizontal_zz),
        abs(right.vertical_zz - left.vertical_zz),
    )


def plane_observables(state, *, environment=None):
    """Estimate bulk observables while growing both dimensions of the plane."""

    if not isinstance(state, UniformPlaneLETTA):
        raise TypeError("state must be a UniformPlaneLETTA.")
    if environment is None:
        environment = PlaneEnvironmentOptions()
    if not isinstance(environment, PlaneEnvironmentOptions):
        raise TypeError("environment must be a PlaneEnvironmentOptions instance.")
    environment = environment.validated()
    identity, x, z = _spin_operators(state.local_physical_dim)
    cells = tuple(
        double_layer_cell(state, operator)
        for operator in (identity, x, z)
    )
    estimates = []
    for size in environment.window_sizes:
        estimate = _estimate_observables(state, size, environment, cells)
        estimates.append(estimate)
    final = estimates[-1]
    window_change = (
        float("inf")
        if len(estimates) < 2
        else _observable_change(estimates[-2], final)
    )
    window_converged = window_change <= environment.convergence_tolerance

    boundary_estimates = [
        _estimate_observables(
            state,
            final.window_size,
            environment,
            cells,
            boundary_bond_dim=boundary_bond_dim,
        )
        for boundary_bond_dim in environment.boundary_bond_dims[:-1]
    ]
    boundary_estimates.append(final)
    if len(boundary_estimates) > 1:
        boundary_change = _observable_change(
            boundary_estimates[-2],
            boundary_estimates[-1],
        )
        boundary_converged = (
            boundary_change
            <= environment.boundary_convergence_tolerance
        )
    else:
        boundary_change = 0.0
        boundary_converged = (
            final.maximum_boundary_bond_dimension
            < environment.boundary_bond_dim
            and final.discarded_weight
            <= environment.boundary_convergence_tolerance
        )
    all_estimates = tuple(estimates) + tuple(boundary_estimates[:-1])
    converged = window_converged and boundary_converged
    return PlaneObservables(
        transverse_magnetization=final.transverse_magnetization,
        horizontal_zz=final.horizontal_zz,
        vertical_zz=final.vertical_zz,
        window_size=final.window_size,
        converged=converged,
        window_converged=window_converged,
        boundary_converged=boundary_converged,
        window_change=window_change,
        boundary_change=boundary_change,
        estimates=tuple(estimates),
        boundary_estimates=tuple(boundary_estimates),
        maximum_boundary_bond_dimension=max(
            estimate.maximum_boundary_bond_dimension
            for estimate in all_estimates
        ),
        discarded_weight=max(
            estimate.discarded_weight for estimate in all_estimates
        ),
    )


def plane_energy_density(state, model, *, environment=None):
    r"""Return the square-lattice TFIM energy per site.

    There is one horizontal and one vertical nearest-neighbor bond per site:
    $e=-J(\langle ZZ\rangle_x+\langle ZZ\rangle_y)-g\langle X\rangle$.
    """

    if not isinstance(model, PlaneTFIM):
        raise TypeError("model must be a PlaneTFIM.")
    observables = plane_observables(state, environment=environment)
    return float(
        -model.coupling
        * (observables.horizontal_zz + observables.vertical_zz)
        - model.field * observables.transverse_magnetization
    )
