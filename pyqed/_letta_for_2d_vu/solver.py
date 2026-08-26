"""Variational uniform LETTA optimization on infinite cylinders."""

from __future__ import annotations

from dataclasses import dataclass

from pyqed._vuletta import (
    VULETTAOptions,
    VULETTAResult,
    vuletta,
)

from .operators import CylinderTFIM, _validate_state_model
from .state import UniformCylinderLETTA


@dataclass(frozen=True)
class CylinderVULETTAIteration:
    iteration: int
    energy_density: float
    energy_density_change: float | None
    residual_norm: float | None
    step_size: float | None


@dataclass(frozen=True)
class CylinderVULETTAResult:
    state: UniformCylinderLETTA
    energy_density: float
    energy_per_column: float
    converged: bool
    iterations: int
    function_evaluations: int
    gradient_norm: float
    coordinate_gradient_norm: float
    parameter_norm: float
    residual_norm: float
    metric_rank: int | None
    update_method: str
    gradient_method: str
    history: tuple[CylinderVULETTAIteration, ...]
    message: str
    raw_result: VULETTAResult


def vuletta_cylinder(
    model,
    *,
    bond_dim=None,
    initial=None,
    seed=None,
    real=True,
    options=None,
):
    """Optimize a column-blocked LETTA directly in the infinite direction."""

    if not isinstance(model, CylinderTFIM):
        raise TypeError("model must be a CylinderTFIM.")
    if initial is not None:
        _validate_state_model(initial, model)
        uniform_initial = initial.uniform_state
        if bond_dim is not None and initial.bond_dim != int(bond_dim):
            raise ValueError("initial state and requested bond_dim do not agree.")
        bond_dim = initial.bond_dim
    else:
        uniform_initial = None
        bond_dim = 2 if bond_dim is None else bond_dim
    if options is None:
        options = VULETTAOptions(
            update_method="lbfgs",
            gradient_method="analytic",
        )
    raw = vuletta(
        model.local_density,
        physical_dim=model.column_dim,
        bond_dim=bond_dim,
        initial=uniform_initial,
        seed=seed,
        real=real,
        options=options,
    )
    state = UniformCylinderLETTA(
        model.width,
        model.local_physical_dim,
        raw.state.tensor,
        model.transverse_boundary,
    )
    history = tuple(
        CylinderVULETTAIteration(
            iteration=record.iteration,
            energy_density=record.energy / model.width,
            energy_density_change=(
                None
                if record.energy_change is None
                else record.energy_change / model.width
            ),
            residual_norm=record.residual_norm,
            step_size=record.step_size,
        )
        for record in raw.history
    )
    return CylinderVULETTAResult(
        state=state,
        energy_density=raw.energy / model.width,
        energy_per_column=raw.energy,
        converged=raw.converged,
        iterations=raw.iterations,
        function_evaluations=raw.function_evaluations,
        gradient_norm=raw.gradient_norm,
        coordinate_gradient_norm=raw.coordinate_gradient_norm,
        parameter_norm=raw.parameter_norm,
        residual_norm=raw.residual_norm,
        metric_rank=raw.metric_rank,
        update_method=raw.update_method,
        gradient_method=raw.gradient_method,
        history=history,
        message=raw.message,
        raw_result=raw,
    )
