"""Uniform column-blocked LETTA states for infinite cylinders."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np

from pyqed._vuletta.state import (
    UniformLETTA,
    expand_uniform_letta,
    random_uniform_letta,
)


def _positive_integer(value, name):
    try:
        value = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer.") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _transverse_boundary(value):
    value = str(value).lower()
    if value not in {"open", "periodic"}:
        raise ValueError("transverse_boundary must be 'open' or 'periodic'.")
    return value


@dataclass(frozen=True)
class UniformCylinderLETTA:
    """A LETTA uniform along the infinite direction of a finite-width strip.

    The two tied physical indices are complete adjacent column
    configurations. The tensor order is
    ``(left_virtual, previous_column, current_column, right_virtual)``.
    """

    width: int
    local_physical_dim: int
    tensor: np.ndarray
    transverse_boundary: str = "periodic"

    def __post_init__(self):
        width = _positive_integer(self.width, "width")
        local_dim = _positive_integer(
            self.local_physical_dim,
            "local_physical_dim",
        )
        boundary = _transverse_boundary(self.transverse_boundary)
        tensor = np.asarray(self.tensor)
        expected_column_dim = local_dim**width
        if tensor.ndim != 4:
            raise ValueError("a cylinder LETTA tensor must have four axes.")
        if tensor.shape[0] != tensor.shape[-1]:
            raise ValueError("left and right LETTA bond dimensions must agree.")
        if tensor.shape[1:3] != (
            expected_column_dim,
            expected_column_dim,
        ):
            raise ValueError(
                "the tied physical dimensions must equal "
                "local_physical_dim**width."
            )
        validated = UniformLETTA(tensor)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "local_physical_dim", local_dim)
        object.__setattr__(self, "transverse_boundary", boundary)
        object.__setattr__(self, "tensor", validated.tensor)

    @property
    def column_dim(self):
        return self.local_physical_dim**self.width

    @property
    def bond_dim(self):
        return int(self.tensor.shape[0])

    @property
    def effective_bond_dim(self):
        return self.bond_dim * self.column_dim

    @property
    def parameter_count(self):
        return int(self.tensor.size)

    @property
    def uniform_state(self):
        return UniformLETTA(self.tensor)

    def normalized_parameters(self):
        tensor = self.tensor / np.linalg.norm(self.tensor)
        return type(self)(
            self.width,
            self.local_physical_dim,
            tensor,
            self.transverse_boundary,
        )

    def encode_column(self, configuration):
        if np.isscalar(configuration):
            value = index(configuration)
            if value < 0 or value >= self.column_dim:
                raise ValueError("column configuration index is out of range.")
            return value
        configuration = tuple(index(value) for value in configuration)
        if len(configuration) != self.width:
            raise ValueError("a column configuration must have width entries.")
        if any(
            value < 0 or value >= self.local_physical_dim
            for value in configuration
        ):
            raise ValueError("a local physical index is out of range.")
        return int(
            np.ravel_multi_index(
                configuration,
                (self.local_physical_dim,) * self.width,
            )
        )

    def decode_column(self, column):
        column = self.encode_column(column)
        return tuple(
            int(value)
            for value in np.unravel_index(
                column,
                (self.local_physical_dim,) * self.width,
            )
        )

    def periodic_amplitude(self, columns):
        encoded = tuple(self.encode_column(column) for column in columns)
        return self.uniform_state.periodic_amplitude(encoded)


def random_uniform_cylinder_letta(
    width,
    *,
    local_physical_dim=2,
    bond_dim=2,
    seed=None,
    real=True,
    transverse_boundary="periodic",
):
    """Return a random normalized uniform cylinder LETTA."""

    width = _positive_integer(width, "width")
    local_dim = _positive_integer(
        local_physical_dim,
        "local_physical_dim",
    )
    boundary = _transverse_boundary(transverse_boundary)
    state = random_uniform_letta(
        local_dim**width,
        bond_dim,
        seed=seed,
        real=real,
    )
    return UniformCylinderLETTA(
        width,
        local_dim,
        state.tensor,
        boundary,
    )


def expand_uniform_cylinder_letta(
    state,
    bond_dim,
    *,
    seed=None,
    relative_noise=3.0e-2,
):
    """Embed a cylinder LETTA into a larger longitudinal virtual space."""

    if not isinstance(state, UniformCylinderLETTA):
        raise TypeError("state must be a UniformCylinderLETTA.")
    expanded = expand_uniform_letta(
        state.uniform_state,
        bond_dim,
        seed=seed,
        relative_noise=relative_noise,
    )
    return UniformCylinderLETTA(
        state.width,
        state.local_physical_dim,
        expanded.tensor,
        state.transverse_boundary,
    )
