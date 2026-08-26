"""Finite-Abelian charge sectors for symmetry-adapted LETTA tensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from operator import index

import numpy as np


def _as_tuple(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


@dataclass(frozen=True)
class AbelianSymmetry:
    r"""A user-defined product of additive Abelian charge factors.

    ``moduli`` defines a product of cyclic groups. A component equal to
    ``None`` is an unrestricted additive integer charge, which is useful for
    particle-number or magnetization sectors. For a single component, scalar
    charges are accepted and returned; product-group charges are tuples.

    The physical basis must diagonalize the symmetry action. LETTA assigns the
    charge of each physical site to the first (owned) physical axis of that
    site's tensor; positive-neighbor axes are dependency indices and do not
    carry the charge a second time.
    """

    physical_charges: tuple
    sector: object
    moduli: object = None
    name: str = "abelian"
    _moduli: tuple[int | None, ...] = field(init=False, repr=False)
    _physical_vectors: tuple[tuple[int, ...], ...] = field(
        init=False, repr=False
    )
    _sector_vector: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self):
        moduli = _as_tuple(self.moduli)
        normalized_moduli = []
        for modulus in moduli:
            if modulus is None:
                normalized_moduli.append(None)
                continue
            try:
                modulus = index(modulus)
            except TypeError as error:
                raise ValueError("symmetry moduli must be integers or None.") from error
            if modulus <= 1:
                raise ValueError("finite symmetry moduli must be greater than one.")
            normalized_moduli.append(modulus)
        moduli = tuple(normalized_moduli)
        physical = tuple(
            self._normalize_vector(charge, moduli)
            for charge in tuple(self.physical_charges)
        )
        if not physical:
            raise ValueError("physical_charges must contain at least one basis charge.")
        sector = self._normalize_vector(self.sector, moduli)
        object.__setattr__(self, "_moduli", moduli)
        object.__setattr__(self, "_physical_vectors", physical)
        object.__setattr__(self, "_sector_vector", sector)
        object.__setattr__(
            self,
            "physical_charges",
            tuple(self._external(vector) for vector in physical),
        )
        object.__setattr__(self, "sector", self._external(sector))
        object.__setattr__(self, "moduli", moduli[0] if len(moduli) == 1 else moduli)
        object.__setattr__(self, "name", str(self.name))

    @staticmethod
    def _normalize_vector(charge, moduli):
        values = _as_tuple(charge)
        if len(values) != len(moduli):
            raise ValueError(
                "every charge must have one component per symmetry modulus."
            )
        normalized = []
        for value, modulus in zip(values, moduli):
            try:
                value = index(value)
            except TypeError as error:
                raise ValueError("symmetry charges must be integers.") from error
            normalized.append(value if modulus is None else value % modulus)
        return tuple(normalized)

    def _external(self, vector):
        vector = tuple(vector)
        return vector[0] if len(vector) == 1 else vector

    @property
    def identity(self):
        return self._external((0,) * len(self._moduli))

    @property
    def number_of_components(self):
        return len(self._moduli)

    def normalize(self, charge):
        return self._external(self._normalize_vector(charge, self._moduli))

    def fuse(self, *charges):
        total = [0] * len(self._moduli)
        for charge in charges:
            vector = self._normalize_vector(charge, self._moduli)
            for component, (value, modulus) in enumerate(
                zip(vector, self._moduli)
            ):
                total[component] += value
                if modulus is not None:
                    total[component] %= modulus
        return self._external(total)

    def difference(self, left, right):
        """Return the charge ``left - right`` in the configured group."""

        left = self._normalize_vector(left, self._moduli)
        right = self._normalize_vector(right, self._moduli)
        result = []
        for lhs, rhs, modulus in zip(left, right, self._moduli):
            value = lhs - rhs
            result.append(value if modulus is None else value % modulus)
        return self._external(result)

    def configuration_charge(self, configuration):
        configuration = tuple(index(value) for value in configuration)
        if any(
            value < 0 or value >= len(self.physical_charges)
            for value in configuration
        ):
            raise ValueError("configuration contains an invalid physical index.")
        return self.fuse(
            *(self.physical_charges[value] for value in configuration)
        )

    def _reachable(self, nsites):
        reachable = [{self.identity}]
        for _ in range(nsites):
            reachable.append(
                {
                    self.fuse(charge, physical)
                    for charge in reachable[-1]
                    for physical in self.physical_charges
                }
            )
        return tuple(reachable)

    def allocate_bond_charges(self, nsites, bond_dim):
        """Allocate deterministic valid charge labels for uniform bond sizes."""

        nsites = index(nsites)
        bond_dim = index(bond_dim)
        if nsites <= 0 or bond_dim <= 0:
            raise ValueError("nsites and bond_dim must be positive.")
        if nsites == 1:
            return ()
        reachable = self._reachable(nsites)
        if self.sector not in reachable[-1]:
            raise ValueError(
                "the requested sector is unreachable from the physical charges."
            )
        valid = []
        for cut in range(1, nsites):
            valid.append(
                {
                    charge
                    for charge in reachable[cut]
                    if self.difference(self.sector, charge)
                    in reachable[nsites - cut]
                }
            )

        witness = []
        current = self.identity
        for position in range(1, nsites + 1):
            candidates = []
            for physical in self.physical_charges:
                proposed = self.fuse(current, physical)
                if self.difference(self.sector, proposed) in reachable[
                    nsites - position
                ]:
                    candidates.append(proposed)
            if not candidates:
                raise ValueError("failed to construct a path to the target sector.")
            current = sorted(candidates, key=repr)[0]
            if position < nsites:
                witness.append(current)

        allocations = []
        for required, available in zip(witness, valid):
            ordered = [required] + sorted(available - {required}, key=repr)
            retained = ordered[:bond_dim]
            allocations.append(
                tuple(retained[position % len(retained)] for position in range(bond_dim))
            )
        return tuple(allocations)

    def validate_bond_charges(self, bond_charges, bond_dimensions):
        bond_charges = tuple(tuple(charges) for charges in bond_charges)
        bond_dimensions = tuple(index(value) for value in bond_dimensions)
        if len(bond_charges) != len(bond_dimensions):
            raise ValueError("there must be one charge list per virtual bond.")
        result = []
        for charges, dimension in zip(bond_charges, bond_dimensions):
            if len(charges) != dimension:
                raise ValueError("a virtual charge list has the wrong dimension.")
            result.append(tuple(self.normalize(charge) for charge in charges))
        return tuple(result)

    def commutation_error(self, operator, *, nsites):
        """Return the largest relative commutator error of the generators."""

        nsites = index(nsites)
        physical_dim = len(self.physical_charges)
        dimension = physical_dim**nsites
        operator = np.asarray(operator)
        if operator.shape != (dimension, dimension):
            raise ValueError("operator shape does not match nsites and physical_dim.")
        configurations = tuple(np.ndindex(*(physical_dim,) * nsites))
        total_vectors = np.asarray(
            [
                self._normalize_vector(
                    self.configuration_charge(configuration), self._moduli
                )
                for configuration in configurations
            ],
            dtype=float,
        )
        scale = max(float(np.linalg.norm(operator)), np.finfo(float).tiny)
        errors = []
        for component, modulus in enumerate(self._moduli):
            values = total_vectors[:, component]
            if modulus is None:
                generator = values
            else:
                generator = np.exp(2j * np.pi * values / modulus)
            commutator = (
                generator[:, None] * operator
                - operator * generator[None, :]
            )
            errors.append(float(np.linalg.norm(commutator) / scale))
        return max(errors, default=0.0)

