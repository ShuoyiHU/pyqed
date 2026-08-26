"""LETTA construction and optimization in the 3D snake ordering."""

from __future__ import annotations

from dataclasses import replace

from .. import (
    LETTADMROptions,
    LatticeLETTA,
    automatic_bond_schedule,
    letta_dmrg,
)

from .geometry import ordered_coordinates, validate_shape


def snake_letta_state(
    lattice_shape,
    *,
    physical_dim=2,
    bond_dim=2,
    seed=None,
    real=True,
    ordering="continuous-snake",
):
    """Return a random 3D LETTA with current, +z, +y, and +x spin legs."""

    shape = validate_shape(lattice_shape)
    return LatticeLETTA.random(
        shape,
        physical_dim=physical_dim,
        bond_dim=bond_dim,
        seed=seed,
        real=real,
        coordinates=ordered_coordinates(shape, ordering=ordering),
    )


def letta_ground_state(
    hamiltonian,
    *,
    lattice_shape,
    bond_dim=2,
    physical_dim=2,
    seed=None,
    state=None,
    options=None,
    use_bond_schedule=False,
    ordering="continuous-snake",
):
    """Optimize a snake-ordered 3D LETTA with the lattice LETTA sweeper."""

    shape = validate_shape(lattice_shape)
    options = LETTADMROptions() if options is None else options
    if not isinstance(options, LETTADMROptions):
        raise TypeError("options must be a LETTADMROptions instance.")
    if state is None:
        initial_bond_dim = bond_dim
        if use_bond_schedule and bond_dim > 2:
            dimensions, sweeps = automatic_bond_schedule(
                bond_dim,
                options.max_sweeps,
            )
            initial_bond_dim = dimensions[0]
            options = replace(
                options,
                bond_dimension_schedule=dimensions,
                bond_schedule_sweeps=sweeps,
            )
        state = snake_letta_state(
            shape,
            physical_dim=physical_dim,
            bond_dim=initial_bond_dim,
            seed=seed,
            ordering=ordering,
        )
    return letta_dmrg(
        hamiltonian,
        state=state,
        physical_dim=physical_dim,
        bond_dim=bond_dim,
        seed=seed,
        options=options,
    )
