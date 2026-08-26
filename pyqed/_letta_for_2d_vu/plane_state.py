"""Uniform leg-tied tensor states on the infinite square lattice."""

from __future__ import annotations

from operator import index

import numpy as np
import opt_einsum as oe


def _positive_integer(value, name):
    try:
        value = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer.") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


class UniformPlaneLETTA:
    r"""A one-site uniform LETTA tensor for the infinite square lattice.

    Tensor axes are ``(left, right, up, down, center, right_spin,
    down_spin)``.  The first four axes are genuine two-dimensional virtual
    bonds.  The last three axes tie the physical spin at a site to the
    tensors immediately to its left and above.
    """

    def __init__(self, tensor):
        tensor = np.asarray(tensor)
        if tensor.ndim != 7:
            raise ValueError("a plane LETTA tensor must have seven axes.")
        if len(set(tensor.shape[:4])) != 1:
            raise ValueError("all four plane virtual bond dimensions must agree.")
        if len(set(tensor.shape[4:])) != 1:
            raise ValueError("all three tied physical dimensions must agree.")
        if tensor.shape[0] <= 0 or tensor.shape[4] <= 0:
            raise ValueError("LETTA dimensions must be positive.")
        if not np.all(np.isfinite(tensor)):
            raise ValueError("LETTA tensor entries must be finite.")
        if np.linalg.norm(tensor) <= np.finfo(float).tiny:
            raise ValueError("a plane LETTA tensor cannot be numerically zero.")
        self.tensor = tensor.copy()

    @property
    def bond_dim(self):
        return int(self.tensor.shape[0])

    @property
    def local_physical_dim(self):
        return int(self.tensor.shape[4])

    @property
    def parameter_count(self):
        return int(self.tensor.size)

    @property
    def double_layer_bond_dim(self):
        return self.bond_dim**2 * self.local_physical_dim**2

    def copy(self):
        return type(self)(self.tensor)

    def normalized_parameters(self):
        return type(self)(self.tensor / np.linalg.norm(self.tensor))

    def periodic_amplitude(self, configuration):
        """Contract the LETTA amplitude on a finite periodic square torus."""

        configuration = np.asarray(configuration)
        if configuration.ndim != 2 or min(configuration.shape) <= 0:
            raise ValueError("configuration must be a nonempty rank-two array.")
        if not np.issubdtype(configuration.dtype, np.integer):
            if not np.all(configuration == np.asarray(configuration, dtype=int)):
                raise ValueError("configuration entries must be integers.")
            configuration = np.asarray(configuration, dtype=int)
        if np.any(configuration < 0) or np.any(
            configuration >= self.local_physical_dim
        ):
            raise ValueError("configuration contains an invalid physical index.")

        height, width = configuration.shape
        horizontal = np.arange(height * width).reshape(height, width)
        vertical = horizontal + height * width
        operands = []
        for row in range(height):
            for column in range(width):
                local = self.tensor[
                    :,
                    :,
                    :,
                    :,
                    configuration[row, column],
                    configuration[row, (column + 1) % width],
                    configuration[(row + 1) % height, column],
                ]
                labels = (
                    int(horizontal[row, (column - 1) % width]),
                    int(horizontal[row, column]),
                    int(vertical[(row - 1) % height, column]),
                    int(vertical[row, column]),
                )
                operands.extend((local, labels))
        return oe.contract(*operands, optimize="auto")


def random_uniform_plane_letta(
    *,
    local_physical_dim=2,
    bond_dim=1,
    seed=None,
    real=True,
):
    """Return a random normalized square-lattice LETTA tensor."""

    local_dim = _positive_integer(local_physical_dim, "local_physical_dim")
    bond_dim = _positive_integer(bond_dim, "bond_dim")
    rng = np.random.default_rng(seed)
    shape = (bond_dim,) * 4 + (local_dim,) * 3
    tensor = rng.normal(size=shape)
    if not real:
        tensor = tensor + 1j * rng.normal(size=shape)
    tensor /= np.linalg.norm(tensor)
    return UniformPlaneLETTA(tensor)


def expand_uniform_plane_letta(
    state,
    bond_dim,
    *,
    seed=None,
    relative_noise=3.0e-2,
):
    """Embed a plane LETTA into a larger four-directional virtual space."""

    if not isinstance(state, UniformPlaneLETTA):
        raise TypeError("state must be a UniformPlaneLETTA.")
    bond_dim = _positive_integer(bond_dim, "bond_dim")
    if bond_dim <= state.bond_dim:
        raise ValueError("the expanded bond dimension must be larger.")
    relative_noise = float(relative_noise)
    if not np.isfinite(relative_noise) or relative_noise <= 0.0:
        raise ValueError("relative_noise must be finite and positive.")

    rng = np.random.default_rng(seed)
    shape = (bond_dim,) * 4 + (state.local_physical_dim,) * 3
    perturbation = rng.normal(size=shape)
    if np.iscomplexobj(state.tensor):
        perturbation = perturbation + 1j * rng.normal(size=shape)
    perturbation /= np.linalg.norm(perturbation)
    tensor = relative_noise * perturbation
    old = state.bond_dim
    retained = (slice(0, old),) * 4 + (slice(None),) * 3
    tensor[retained] += state.tensor / np.linalg.norm(state.tensor)
    return UniformPlaneLETTA(tensor / np.linalg.norm(tensor))
