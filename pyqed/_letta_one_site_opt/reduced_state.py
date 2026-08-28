"""Exact reduced-SU(2) LETTA state storage and small-system reconstruction."""

from __future__ import annotations

from collections import Counter
from operator import index

import numpy as np

from pyqed.mps.nonabelian.coupling import clebsch_gordan, ordered_two_m_values

from .reduced_symmetry import ReducedBasisState, ReducedSymmetry, _sector_irrep
from .state import _validate_coordinates, _validate_lattice_shape


def _multiplicities(sectors):
    return Counter(tuple(sectors))


class ReducedLatticeLETTA:
    r"""LETTA whose non-scalar SU(2) coordinates are Wigner-Eckart reduced.

    Blocks are keyed by ``(q_left, q_physical, q_right)`` and shaped as

    ``(mult_left, mult_physical, *scalar_condition_axes, mult_right)``.

    The repeated dependency axes contain only reduced irrep/multiplicity labels.
    Magnetic components are reconstructed from Clebsch-Gordan coefficients and
    never stored as variational parameters.
    """

    def __init__(
        self,
        lattice_shape,
        symmetry,
        tensors,
        *,
        bond_sectors,
        coordinates=None,
        normalize=True,
    ):
        self.lattice_shape = _validate_lattice_shape(lattice_shape)
        if not isinstance(symmetry, ReducedSymmetry):
            raise TypeError("symmetry must be a ReducedSymmetry")
        self.symmetry = symmetry
        self.physical_basis = symmetry.physical_basis
        self.coordinates = _validate_coordinates(self.lattice_shape, coordinates)
        self._coordinate_to_site = {
            coordinate: site for site, coordinate in enumerate(self.coordinates)
        }
        self._neighborhoods = tuple(
            self._build_neighborhood(coordinate) for coordinate in self.coordinates
        )
        bond_sectors = tuple(tuple(bond) for bond in bond_sectors)
        if len(bond_sectors) != self.nsites - 1:
            raise ValueError("bond_sectors must contain one entry per internal bond")
        if any(not bond for bond in bond_sectors):
            raise ValueError("every internal bond needs at least one reduced multiplet")
        self.bond_sectors = bond_sectors
        self.tensors = self._validate_tensors(tensors)
        if normalize:
            self.normalize()

    @classmethod
    def random(
        cls,
        lattice_shape,
        *,
        symmetry,
        multiplets_per_sector=1,
        seed=None,
        real=True,
        coordinates=None,
        normalize=True,
    ):
        lattice_shape = _validate_lattice_shape(lattice_shape)
        coordinates = _validate_coordinates(lattice_shape, coordinates)
        nsites = len(coordinates)
        bond_sectors = symmetry.allocate_bond_sectors(
            nsites,
            multiplets_per_sector=multiplets_per_sector,
        )
        coordinate_to_site = {
            coordinate: site for site, coordinate in enumerate(coordinates)
        }
        neighborhoods = []
        for coordinate in coordinates:
            sites = [coordinate_to_site[coordinate]]
            for axis in reversed(range(len(lattice_shape))):
                neighbor = list(coordinate)
                neighbor[axis] += 1
                neighbor = tuple(neighbor)
                if neighbor in coordinate_to_site:
                    sites.append(coordinate_to_site[neighbor])
            neighborhoods.append(tuple(sites))

        rng = np.random.default_rng(seed)
        tensors = []
        for site in range(nsites):
            left_sectors = (
                (symmetry.identity,) if site == 0 else bond_sectors[site - 1]
            )
            right_sectors = (
                (symmetry.sector,) if site == nsites - 1 else bond_sectors[site]
            )
            left_mult = _multiplicities(left_sectors)
            right_mult = _multiplicities(right_sectors)
            dependency_shape = (symmetry.physical_basis.reduced_dim,) * (
                len(neighborhoods[site]) - 1
            )
            blocks = {}
            for q_left in sorted(left_mult):
                for q_phys, d_phys in zip(
                    symmetry.physical_basis.sectors,
                    symmetry.physical_basis.multiplicities,
                ):
                    fused = symmetry.fuse(q_left, q_phys)
                    for q_right in sorted(right_mult):
                        if q_right not in fused:
                            continue
                        shape = (
                            left_mult[q_left],
                            d_phys,
                        ) + dependency_shape + (right_mult[q_right],)
                        block = rng.normal(size=shape)
                        if not real:
                            block = block + 1j * rng.normal(size=shape)
                        block = block / np.sqrt(max(int(np.prod(shape)), 1))
                        blocks[(q_left, q_phys, q_right)] = block
            tensors.append(blocks)
        return cls(
            lattice_shape,
            symmetry,
            tensors,
            bond_sectors=bond_sectors,
            coordinates=coordinates,
            normalize=normalize,
        )

    @property
    def ndim(self):
        return len(self.lattice_shape)

    @property
    def nsites(self):
        return len(self.coordinates)

    @property
    def physical_dim(self):
        """Number of scalar reduced physical labels."""

        return self.physical_basis.reduced_dim

    @property
    def dense_physical_dim(self):
        return self.physical_basis.dense_dim

    @property
    def parameter_count(self):
        return sum(block.size for tensor in self.tensors for block in tensor.values())

    @property
    def target_two_j(self):
        return _sector_irrep(self.symmetry.sector).two_j

    def _build_neighborhood(self, coordinate):
        sites = [self._coordinate_to_site[coordinate]]
        for axis in reversed(range(self.ndim)):
            neighbor = list(coordinate)
            neighbor[axis] += 1
            neighbor = tuple(neighbor)
            if neighbor in self._coordinate_to_site:
                sites.append(self._coordinate_to_site[neighbor])
        return tuple(sites)

    def site_neighborhood(self, site):
        site = index(site)
        if site < 0 or site >= self.nsites:
            raise IndexError("site index out of range")
        return self._neighborhoods[site]

    def left_virtual_sectors(self, site):
        return (self.symmetry.identity,) if site == 0 else self.bond_sectors[site - 1]

    def right_virtual_sectors(self, site):
        return (self.symmetry.sector,) if site == self.nsites - 1 else self.bond_sectors[site]

    def _validate_tensors(self, tensors):
        tensors = tuple(dict(tensor) for tensor in tensors)
        if len(tensors) != self.nsites:
            raise ValueError("there must be one reduced LETTA tensor per lattice site")
        validated = []
        physical_mult = dict(
            zip(self.physical_basis.sectors, self.physical_basis.multiplicities)
        )
        for site, tensor in enumerate(tensors):
            left_mult = _multiplicities(self.left_virtual_sectors(site))
            right_mult = _multiplicities(self.right_virtual_sectors(site))
            dependency_shape = (self.physical_basis.reduced_dim,) * (
                len(self._neighborhoods[site]) - 1
            )
            blocks = {}
            for raw_key, raw_block in tensor.items():
                if len(raw_key) != 3:
                    raise ValueError("reduced LETTA block keys must be (left, physical, right)")
                q_left, q_phys, q_right = raw_key
                if (
                    q_left not in left_mult
                    or q_phys not in physical_mult
                    or q_right not in right_mult
                    or q_right not in self.symmetry.fuse(q_left, q_phys)
                ):
                    raise ValueError(f"forbidden fusion block at site {site}: {raw_key!r}")
                expected = (
                    left_mult[q_left],
                    physical_mult[q_phys],
                ) + dependency_shape + (right_mult[q_right],)
                block = np.asarray(raw_block).copy()
                if block.shape != expected:
                    raise ValueError(
                        f"reduced LETTA block {raw_key!r} at site {site} has shape "
                        f"{block.shape}, expected {expected}"
                    )
                if not np.all(np.isfinite(block)):
                    raise ValueError("reduced LETTA blocks must contain finite values")
                blocks[tuple(raw_key)] = block
            if not blocks:
                raise ValueError(f"reduced LETTA tensor {site} has no fusion-allowed blocks")
            validated.append(blocks)
        return validated

    def copy(self):
        return ReducedLatticeLETTA(
            self.lattice_shape,
            self.symmetry,
            tuple(
                {key: block.copy() for key, block in tensor.items()}
                for tensor in self.tensors
            ),
            bond_sectors=self.bond_sectors,
            coordinates=self.coordinates,
            normalize=False,
        )

    def symmetry_violation(self):
        """Structural reduced storage has no forbidden magnetic components."""

        return 0.0

    @property
    def dense_basis_states(self):
        return tuple(
            (state, two_m)
            for state in self.physical_basis.reduced_states
            for two_m in ordered_two_m_values(state.irrep)
        )

    def state_vector(self, *, target_two_m=None, max_sites=14):
        """Expand one target multiplet component for verification-sized systems."""

        if self.nsites > int(max_sites):
            raise ValueError(
                "dense reduced-LETTA reconstruction is verification-only; "
                "use frontier contractions for larger systems"
            )
        target_irrep = _sector_irrep(self.symmetry.sector)
        if target_two_m is None:
            target_two_m = target_irrep.two_j
        target_two_m = int(target_two_m)
        if target_two_m not in ordered_two_m_values(target_irrep):
            raise ValueError("target_two_m is not a component of the target irrep")

        dense_states = self.dense_basis_states
        dimension = len(dense_states) ** self.nsites
        vector = np.zeros(dimension, dtype=np.result_type(
            complex,
            *[block.dtype for tensor in self.tensors for block in tensor.values()],
        ))
        dimensions = (len(dense_states),) * self.nsites
        for flat, dense_configuration in enumerate(np.ndindex(*dimensions)):
            local_states = tuple(dense_states[value] for value in dense_configuration)
            reduced_configuration = tuple(
                self.physical_basis.condition_index(state)
                for state, _two_m in local_states
            )
            boundary = {(self.symmetry.identity, 0, 0): 1.0 + 0.0j}
            for site, tensor in enumerate(self.tensors):
                physical_state, two_m_phys = local_states[site]
                dependencies = tuple(
                    reduced_configuration[neighbor]
                    for neighbor in self._neighborhoods[site][1:]
                )
                updated = {}
                for (q_left, left_slot, two_m_left), amplitude in boundary.items():
                    for (block_left, block_phys, block_right), block in tensor.items():
                        if block_left != q_left or block_phys != physical_state.sector:
                            continue
                        matrix = block[
                            (slice(None), physical_state.copy)
                            + dependencies
                            + (slice(None),)
                        ]
                        if left_slot >= matrix.shape[0]:
                            continue
                        for right_slot in range(matrix.shape[1]):
                            reduced_value = matrix[left_slot, right_slot]
                            if reduced_value == 0:
                                continue
                            for two_m_right in ordered_two_m_values(
                                _sector_irrep(block_right)
                            ):
                                coeff = clebsch_gordan(
                                    _sector_irrep(block_left),
                                    _sector_irrep(block_phys),
                                    _sector_irrep(block_right),
                                    two_m_left,
                                    two_m_phys,
                                    two_m_right,
                                )
                                if coeff:
                                    key = (block_right, right_slot, two_m_right)
                                    updated[key] = updated.get(key, 0.0) + (
                                        amplitude * reduced_value * coeff
                                    )
                boundary = updated
            vector[flat] = boundary.get(
                (self.symmetry.sector, 0, target_two_m),
                0.0,
            )
        if np.max(np.abs(vector.imag), initial=0.0) <= 1.0e-13:
            return vector.real
        return vector

    def norm(self):
        # Contract only local CG component spaces. This is polynomial in the
        # explicit frontier dimensions and avoids the full state vector.
        from .reduced_contraction import (
            CanonicalEnvironmentChain,
            identity_canonical_factors,
        )
        from .reduced_frontier import ReducedFrontier

        sites = tuple(ReducedFrontier.from_state(self).to_mps(self))
        value = CanonicalEnvironmentChain.build(
            sites, identity_canonical_factors(sites)
        ).expectation()
        # The open right boundary contains the complete target multiplet, so
        # the invariant contraction sums identical norms over all of its
        # magnetic components.  ``state_vector`` and the optimizer use one
        # normalized target-M component as their state convention.
        target_dimension = self.target_two_j + 1
        return float(np.real(value)) / target_dimension

    def balance_scalar_gauge(self):
        """Equalize site Frobenius norms without changing the represented state.

        Multiplying every block at site ``i`` by a scalar ``s_i`` changes no
        amplitude when ``prod_i s_i = 1``.  Keeping those scalar gauges near
        one prevents otherwise harmless local-solver rescalings from causing
        cancellation in long environment contractions.
        """

        norms = np.asarray(
            [
                np.sqrt(
                    sum(
                        float(np.real(np.vdot(block, block)))
                        for block in tensor.values()
                    )
                )
                for tensor in self.tensors
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(norms)) or np.any(norms <= np.finfo(float).tiny):
            raise ValueError("cannot balance a zero or non-finite reduced LETTA tensor")
        if self.nsites <= 1:
            return self
        log_norms = np.log(norms)
        log_scales = np.mean(log_norms) - log_norms
        # Enforce the state-preserving product exactly in log arithmetic.
        log_scales[-1] = -np.sum(log_scales[:-1])
        scales = np.exp(log_scales)
        if np.any(~np.isfinite(scales)):
            raise ValueError("reduced LETTA scalar gauge is too ill-conditioned")
        for tensor, scale in zip(self.tensors, scales):
            for key in tuple(tensor):
                tensor[key] = tensor[key] * scale
        return self

    def normalize(self):
        norm_squared = self.norm()
        if norm_squared <= np.finfo(float).tiny:
            raise ValueError("cannot normalize a numerically zero reduced LETTA state")
        scale = norm_squared ** -0.5
        for key in tuple(self.tensors[0]):
            self.tensors[0][key] = self.tensors[0][key] * scale
        return self.balance_scalar_gauge()


__all__ = ["ReducedLatticeLETTA"]
