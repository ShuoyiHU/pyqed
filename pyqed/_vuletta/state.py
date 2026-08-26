"""Uniform nearest-neighbor leg-tied tensor states."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np


def _as_letta_tensor(tensor):
    array = np.asarray(tensor)
    if array.ndim != 4:
        raise ValueError(
            "a uniform LETTA tensor must have shape "
            "(left, previous_physical, current_physical, right)."
        )
    if array.shape[0] != array.shape[3]:
        raise ValueError("left and right LETTA bond dimensions must agree.")
    if array.shape[1] != array.shape[2]:
        raise ValueError("the two shared physical dimensions must agree.")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("bond and physical dimensions must be positive.")
    if not np.all(np.isfinite(array)):
        raise ValueError("a LETTA tensor must contain only finite values.")
    if np.linalg.norm(array) <= np.finfo(float).tiny:
        raise ValueError("a LETTA tensor cannot be numerically zero.")
    return array


@dataclass(frozen=True)
class UniformLETTA:
    """One-site uniform nearest-neighbor LETTA state.

    The tensor order is ``(left, previous_physical, current_physical, right)``.
    The periodic amplitude is ``trace(prod_n T[s_n, s_(n+1)]))``.
    """

    tensor: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "tensor", _as_letta_tensor(self.tensor))

    @property
    def bond_dim(self):
        return int(self.tensor.shape[0])

    @property
    def physical_dim(self):
        return int(self.tensor.shape[1])

    @property
    def effective_bond_dim(self):
        return self.bond_dim * self.physical_dim

    def normalized_parameters(self):
        """Return a copy with unit Frobenius norm."""

        return UniformLETTA(self.tensor / np.linalg.norm(self.tensor))

    def periodic_amplitude(self, configuration):
        """Return the amplitude of a finite periodic configuration."""

        configuration = tuple(int(value) for value in configuration)
        if not configuration:
            raise ValueError("a periodic configuration must contain at least one site.")
        if any(value < 0 or value >= self.physical_dim for value in configuration):
            raise ValueError("a physical configuration index is out of range.")
        product = np.eye(self.bond_dim, dtype=self.tensor.dtype)
        for site, physical in enumerate(configuration):
            neighbor = configuration[(site + 1) % len(configuration)]
            product = product @ self.tensor[:, physical, neighbor, :]
        return np.trace(product)

    def structured_mps_tensor(self):
        """Return the exact sparse MPS contraction identity for this LETTA."""

        bond = self.bond_dim
        physical_dim = self.physical_dim
        effective_bond = self.effective_bond_dim
        structured = np.zeros(
            (effective_bond, physical_dim, effective_bond),
            dtype=self.tensor.dtype,
        )
        for left in range(bond):
            for previous in range(physical_dim):
                left_combined = left * physical_dim + previous
                for current in range(physical_dim):
                    for right in range(bond):
                        right_combined = right * physical_dim + current
                        structured[left_combined, current, right_combined] = (
                            self.tensor[left, previous, current, right]
                        )
        return structured

    def gauge_transform(self, gauges):
        """Apply ``T[p,s] -> inv(G[p]) T[p,s] G[s]``."""

        gauges = np.asarray(gauges)
        expected = (self.physical_dim, self.bond_dim, self.bond_dim)
        if gauges.shape != expected:
            raise ValueError(f"gauges must have shape {expected}.")
        transformed = np.empty(
            self.tensor.shape,
            dtype=np.result_type(self.tensor.dtype, gauges.dtype),
        )
        for previous in range(self.physical_dim):
            for current in range(self.physical_dim):
                transformed[:, previous, current, :] = np.linalg.solve(
                    gauges[previous],
                    self.tensor[:, previous, current, :] @ gauges[current],
                )
        return UniformLETTA(transformed)


def random_uniform_letta(physical_dim, bond_dim, *, seed=None, real=False):
    """Return a random normalized uniform LETTA tensor."""

    try:
        physical_dim = index(physical_dim)
        bond_dim = index(bond_dim)
    except TypeError as error:
        raise ValueError("physical_dim and bond_dim must be integers.") from error
    if physical_dim <= 0 or bond_dim <= 0:
        raise ValueError("physical_dim and bond_dim must be positive.")
    rng = np.random.default_rng(seed)
    shape = (bond_dim, physical_dim, physical_dim, bond_dim)
    tensor = rng.normal(size=shape)
    if not real:
        tensor = tensor + 1j * rng.normal(size=shape)
    tensor /= np.linalg.norm(tensor)
    return UniformLETTA(tensor)


def expand_uniform_letta(
    state,
    bond_dim,
    *,
    seed=None,
    relative_noise=3.0e-2,
):
    """Embed a converged LETTA tensor into a larger virtual space.

    A normalized perturbation activates the added virtual sector and avoids a
    rank-deficient transfer operator at the exactly block-padded tensor.
    """

    if not isinstance(state, UniformLETTA):
        raise TypeError("state must be a UniformLETTA.")
    try:
        bond_dim = index(bond_dim)
    except TypeError as error:
        raise ValueError("bond_dim must be an integer.") from error
    if bond_dim <= state.bond_dim:
        raise ValueError("the expanded bond dimension must be larger.")
    relative_noise = float(relative_noise)
    if not np.isfinite(relative_noise) or relative_noise <= 0.0:
        raise ValueError("relative_noise must be finite and positive.")

    rng = np.random.default_rng(seed)
    shape = (
        bond_dim,
        state.physical_dim,
        state.physical_dim,
        bond_dim,
    )
    perturbation = rng.normal(size=shape)
    if np.iscomplexobj(state.tensor):
        perturbation = perturbation + 1j * rng.normal(size=shape)
    perturbation /= np.linalg.norm(perturbation)
    tensor = relative_noise * perturbation
    old_bond_dim = state.bond_dim
    tensor[:old_bond_dim, :, :, :old_bond_dim] += (
        state.tensor / np.linalg.norm(state.tensor)
    )
    return UniformLETTA(tensor / np.linalg.norm(tensor))
