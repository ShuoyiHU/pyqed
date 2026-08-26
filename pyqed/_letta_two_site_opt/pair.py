"""Shared-physical-index-aware algebra for adjacent LETTA tensors."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np

from .._letta_one_site_opt.state import LatticeLETTA


@dataclass(frozen=True)
class LETTAPairLayout:
    """Axis layout for two adjacent LETTA tensors.

    The merged tensor orders physical axes as left-only, shared, and
    right-only. A shared physical site occurs once in the merged tensor even
    though it occurs in both input tensors.
    """

    left_site: int
    left_neighborhood: tuple[int, ...]
    right_neighborhood: tuple[int, ...]
    left_only: tuple[int, ...]
    shared: tuple[int, ...]
    right_only: tuple[int, ...]
    physical_dim: int
    left_shape: tuple[int, ...]
    right_shape: tuple[int, ...]
    symmetry: object = None
    left_virtual_charges: tuple | None = None
    middle_virtual_charges: tuple | None = None
    right_virtual_charges: tuple | None = None

    @classmethod
    def from_state(cls, state, left_site):
        if not isinstance(state, LatticeLETTA):
            raise TypeError("state must be a LatticeLETTA.")
        left_site = index(left_site)
        if left_site < 0 or left_site + 1 >= state.nsites:
            raise IndexError("a LETTA pair must start before the last site.")
        left_neighborhood = state.site_neighborhood(left_site)
        right_neighborhood = state.site_neighborhood(left_site + 1)
        right_set = set(right_neighborhood)
        left_set = set(left_neighborhood)
        shared = tuple(site for site in left_neighborhood if site in right_set)
        left_only = tuple(
            site for site in left_neighborhood if site not in right_set
        )
        right_only = tuple(
            site for site in right_neighborhood if site not in left_set
        )
        return cls(
            left_site=left_site,
            left_neighborhood=left_neighborhood,
            right_neighborhood=right_neighborhood,
            left_only=left_only,
            shared=shared,
            right_only=right_only,
            physical_dim=state.physical_dim,
            left_shape=state.tensors[left_site].shape,
            right_shape=state.tensors[left_site + 1].shape,
            symmetry=state.symmetry,
            left_virtual_charges=(
                state.left_virtual_charges(left_site)
                if state.symmetry is not None
                else None
            ),
            middle_virtual_charges=(
                state.right_virtual_charges(left_site)
                if state.symmetry is not None
                else None
            ),
            right_virtual_charges=(
                state.right_virtual_charges(left_site + 1)
                if state.symmetry is not None
                else None
            ),
        )

    @property
    def sites(self):
        return self.left_site, self.left_site + 1

    @property
    def merged_physical_sites(self):
        return self.left_only + self.shared + self.right_only

    @property
    def merged_shape(self):
        return (
            (self.left_shape[0],)
            + (self.physical_dim,) * len(self.merged_physical_sites)
            + (self.right_shape[-1],)
        )

    def factor_mask(self, side):
        """Return the charge-conserving mask for one factor of this pair."""

        if side == "left":
            shape = self.left_shape
            site = self.left_site
            neighborhood = self.left_neighborhood
            left_charges = self.left_virtual_charges
            right_charges = self.middle_virtual_charges
        elif side == "right":
            shape = self.right_shape
            site = self.left_site + 1
            neighborhood = self.right_neighborhood
            left_charges = self.middle_virtual_charges
            right_charges = self.right_virtual_charges
        else:
            raise ValueError("side must be 'left' or 'right'.")
        if self.symmetry is None:
            return np.ones(shape, dtype=bool)
        center_axis = 1 + neighborhood.index(site)
        mask = np.zeros(shape, dtype=bool)
        for left, left_charge in enumerate(left_charges):
            for physical, physical_charge in enumerate(
                self.symmetry.physical_charges
            ):
                outgoing = self.symmetry.fuse(left_charge, physical_charge)
                for right, right_charge in enumerate(right_charges):
                    if outgoing != right_charge:
                        continue
                    target = [slice(None)] * len(shape)
                    target[0] = left
                    target[center_axis] = physical
                    target[-1] = right
                    mask[tuple(target)] = True
        return mask

    def symmetry_mask(self):
        """Return the fixed-total-charge mask of the merged pair tensor."""

        if self.symmetry is None:
            return np.ones(self.merged_shape, dtype=bool)
        left_axis = 1 + self.merged_physical_sites.index(self.left_site)
        right_axis = 1 + self.merged_physical_sites.index(self.left_site + 1)
        mask = np.zeros(self.merged_shape, dtype=bool)
        for left, left_charge in enumerate(self.left_virtual_charges):
            for left_physical, left_charge_physical in enumerate(
                self.symmetry.physical_charges
            ):
                middle = self.symmetry.fuse(
                    left_charge, left_charge_physical
                )
                for right_physical, right_charge_physical in enumerate(
                    self.symmetry.physical_charges
                ):
                    outgoing = self.symmetry.fuse(
                        middle, right_charge_physical
                    )
                    for right, right_charge in enumerate(
                        self.right_virtual_charges
                    ):
                        if outgoing != right_charge:
                            continue
                        target = [slice(None)] * len(self.merged_shape)
                        target[0] = left
                        target[left_axis] = left_physical
                        target[right_axis] = right_physical
                        target[-1] = right
                        mask[tuple(target)] = True
        return mask

    def symmetry_indices(self):
        return np.flatnonzero(self.symmetry_mask().reshape(-1))

    def _validate_tensor_shapes(self, left_tensor, right_tensor):
        expected_left = self.left_shape[:-1]
        expected_right = self.right_shape[1:]
        if tuple(left_tensor.shape[:-1]) != expected_left:
            raise ValueError("left tensor shape does not match the pair layout.")
        if tuple(right_tensor.shape[1:]) != expected_right:
            raise ValueError("right tensor shape does not match the pair layout.")
        if left_tensor.shape[-1] != right_tensor.shape[0]:
            raise ValueError("the pair tensors have incompatible virtual bonds.")

    def _contraction_labels(self):
        next_label = 0

        def label():
            nonlocal next_label
            result = next_label
            next_label += 1
            return result

        left_virtual = label()
        physical = {
            site: label() for site in self.merged_physical_sites
        }
        middle_virtual = label()
        right_virtual = label()
        left_labels = (
            [left_virtual]
            + [physical[site] for site in self.left_neighborhood]
            + [middle_virtual]
        )
        right_labels = (
            [middle_virtual]
            + [physical[site] for site in self.right_neighborhood]
            + [right_virtual]
        )
        merged_labels = (
            [left_virtual]
            + [physical[site] for site in self.merged_physical_sites]
            + [right_virtual]
        )
        return left_labels, right_labels, merged_labels

    def merge(self, left_tensor, right_tensor):
        """Contract the virtual bond and identify shared physical axes."""

        left_tensor = np.asarray(left_tensor)
        right_tensor = np.asarray(right_tensor)
        self._validate_tensor_shapes(left_tensor, right_tensor)

        left_labels, right_labels, output_labels = self._contraction_labels()
        return np.einsum(
            left_tensor,
            left_labels,
            right_tensor,
            right_labels,
            output_labels,
            optimize=True,
        )

    def left_adjoint(self, merged_gradient, right_tensor):
        """Apply the adjoint of the merge map with the right factor fixed."""

        merged_gradient = np.asarray(merged_gradient)
        right_tensor = np.asarray(right_tensor)
        if tuple(merged_gradient.shape) != self.merged_shape:
            raise ValueError("merged gradient shape does not match the pair layout.")
        if tuple(right_tensor.shape[1:]) != self.right_shape[1:]:
            raise ValueError("right tensor shape does not match the pair layout.")
        left_labels, right_labels, merged_labels = self._contraction_labels()
        return np.einsum(
            merged_gradient,
            merged_labels,
            right_tensor.conj(),
            right_labels,
            left_labels,
            optimize=True,
        )

    def right_adjoint(self, left_tensor, merged_gradient):
        """Apply the adjoint of the merge map with the left factor fixed."""

        left_tensor = np.asarray(left_tensor)
        merged_gradient = np.asarray(merged_gradient)
        if tuple(left_tensor.shape[:-1]) != self.left_shape[:-1]:
            raise ValueError("left tensor shape does not match the pair layout.")
        if tuple(merged_gradient.shape) != self.merged_shape:
            raise ValueError("merged gradient shape does not match the pair layout.")
        left_labels, right_labels, merged_labels = self._contraction_labels()
        return np.einsum(
            left_tensor.conj(),
            left_labels,
            merged_gradient,
            merged_labels,
            right_labels,
            optimize=True,
        )


@dataclass(frozen=True)
class LETTASplit:
    left_tensor: np.ndarray
    right_tensor: np.ndarray
    discarded_weight: float
    sector_ranks: tuple[int, ...]


def _tensor_to_neighborhood_order(tensor, canonical_sites, neighborhood):
    permutation = (
        [0]
        + [1 + canonical_sites.index(site) for site in neighborhood]
        + [tensor.ndim - 1]
    )
    return tensor.transpose(permutation)


def conditional_svd_split(
    merged,
    layout,
    *,
    max_bond_dim,
    direction,
    cutoff=0.0,
):
    """Split a merged pair independently for every shared configuration."""

    if not isinstance(layout, LETTAPairLayout):
        raise TypeError("layout must be a LETTAPairLayout.")
    merged = np.asarray(merged)
    if tuple(merged.shape) != layout.merged_shape:
        raise ValueError("merged tensor shape does not match the pair layout.")
    max_bond_dim = index(max_bond_dim)
    if max_bond_dim <= 0:
        raise ValueError("max_bond_dim must be positive.")
    cutoff = float(cutoff)
    if cutoff < 0.0:
        raise ValueError("cutoff must be nonnegative.")
    direction = str(direction).lower()
    if direction not in {"lr", "rl"}:
        raise ValueError("direction must be 'lr' or 'rl'.")

    if layout.symmetry is not None:
        return _symmetry_conditional_svd_split(
            merged,
            layout,
            max_bond_dim=max_bond_dim,
            direction=direction,
            cutoff=cutoff,
        )

    physical_dim = layout.physical_dim
    left_count = len(layout.left_only)
    shared_count = len(layout.shared)
    right_count = len(layout.right_only)
    left_dim = merged.shape[0]
    right_dim = merged.shape[-1]
    dtype = merged.dtype
    left_canonical = np.zeros(
        (left_dim,)
        + (physical_dim,) * (left_count + shared_count)
        + (max_bond_dim,),
        dtype=dtype,
    )
    right_canonical = np.zeros(
        (max_bond_dim,)
        + (physical_dim,) * (shared_count + right_count)
        + (right_dim,),
        dtype=dtype,
    )

    sector_ranks = []
    discarded = 0.0
    total = 0.0
    shared_configurations = np.ndindex(
        *((physical_dim,) * shared_count)
    )
    for shared_configuration in shared_configurations:
        merged_source = (
            (slice(None),)
            + (slice(None),) * left_count
            + shared_configuration
            + (slice(None),) * right_count
            + (slice(None),)
        )
        sector = merged[merged_source]
        matrix = sector.reshape(
            left_dim * physical_dim**left_count,
            physical_dim**right_count * right_dim,
        )
        left_vectors, singular_values, right_vectors = np.linalg.svd(
            matrix,
            full_matrices=False,
        )
        threshold = cutoff * singular_values[0] if singular_values.size else 0.0
        available = int(np.count_nonzero(singular_values > threshold))
        keep = min(max_bond_dim, available)
        sector_ranks.append(keep)
        total += float(np.sum(singular_values**2))
        discarded += float(np.sum(singular_values[keep:] ** 2))
        if keep == 0:
            continue

        left_factor = left_vectors[:, :keep]
        right_factor = right_vectors[:keep]
        retained_values = singular_values[:keep]
        if direction == "lr":
            right_factor = retained_values[:, None] * right_factor
        else:
            left_factor = left_factor * retained_values[None, :]

        left_value = left_factor.reshape(
            (left_dim,) + (physical_dim,) * left_count + (keep,)
        )
        right_value = right_factor.reshape(
            (keep,) + (physical_dim,) * right_count + (right_dim,)
        )
        left_target = (
            (slice(None),)
            + (slice(None),) * left_count
            + shared_configuration
            + (slice(0, keep),)
        )
        right_target = (
            (slice(0, keep),)
            + shared_configuration
            + (slice(None),) * right_count
            + (slice(None),)
        )
        left_canonical[left_target] = left_value
        right_canonical[right_target] = right_value

    relative_discarded = discarded / total if total > 0.0 else 0.0
    left_tensor = _tensor_to_neighborhood_order(
        left_canonical,
        layout.left_only + layout.shared,
        layout.left_neighborhood,
    )
    right_tensor = _tensor_to_neighborhood_order(
        right_canonical,
        layout.shared + layout.right_only,
        layout.right_neighborhood,
    )
    return LETTASplit(
        left_tensor=left_tensor,
        right_tensor=right_tensor,
        discarded_weight=relative_discarded,
        sector_ranks=tuple(sector_ranks),
    )


def _symmetry_conditional_svd_split(
    merged,
    layout,
    *,
    max_bond_dim,
    direction,
    cutoff,
):
    """Split every shared configuration and virtual charge block."""

    if max_bond_dim != len(layout.middle_virtual_charges):
        raise ValueError(
            "max_bond_dim must match the symmetry charge allocation on the bond."
        )
    physical_dim = layout.physical_dim
    left_count = len(layout.left_only)
    shared_count = len(layout.shared)
    right_count = len(layout.right_only)
    left_dim = merged.shape[0]
    right_dim = merged.shape[-1]
    dtype = merged.dtype
    left_canonical = np.zeros(
        (left_dim,)
        + (physical_dim,) * (left_count + shared_count)
        + (max_bond_dim,),
        dtype=dtype,
    )
    right_canonical = np.zeros(
        (max_bond_dim,)
        + (physical_dim,) * (shared_count + right_count)
        + (right_dim,),
        dtype=dtype,
    )
    middle_groups = {}
    for position, charge in enumerate(layout.middle_virtual_charges):
        middle_groups.setdefault(charge, []).append(position)
    left_center_shared = (
        layout.shared.index(layout.left_site)
        if layout.left_site in layout.shared
        else None
    )
    left_center_only = (
        layout.left_only.index(layout.left_site)
        if layout.left_site in layout.left_only
        else None
    )
    right_site = layout.left_site + 1
    right_center_shared = (
        layout.shared.index(right_site) if right_site in layout.shared else None
    )
    right_center_only = (
        layout.right_only.index(right_site)
        if right_site in layout.right_only
        else None
    )
    left_configurations = tuple(np.ndindex(*((physical_dim,) * left_count)))
    right_configurations = tuple(np.ndindex(*((physical_dim,) * right_count)))
    sector_ranks = []
    discarded = 0.0
    total = 0.0

    for shared_configuration in np.ndindex(*((physical_dim,) * shared_count)):
        merged_source = (
            (slice(None),)
            + (slice(None),) * left_count
            + shared_configuration
            + (slice(None),) * right_count
            + (slice(None),)
        )
        matrix = merged[merged_source].reshape(
            left_dim * physical_dim**left_count,
            physical_dim**right_count * right_dim,
        )
        left_matrix = np.zeros((matrix.shape[0], max_bond_dim), dtype=dtype)
        right_matrix = np.zeros((max_bond_dim, matrix.shape[1]), dtype=dtype)
        retained_in_shared = 0

        for middle_charge, middle_positions in middle_groups.items():
            rows = []
            for left_virtual, left_charge in enumerate(
                layout.left_virtual_charges
            ):
                for configuration_flat, configuration in enumerate(
                    left_configurations
                ):
                    if left_center_shared is not None:
                        physical = shared_configuration[left_center_shared]
                    else:
                        physical = configuration[left_center_only]
                    physical_charge = layout.symmetry.physical_charges[physical]
                    if (
                        layout.symmetry.fuse(left_charge, physical_charge)
                        == middle_charge
                    ):
                        rows.append(
                            left_virtual * physical_dim**left_count
                            + configuration_flat
                        )
            columns = []
            for configuration_flat, configuration in enumerate(
                right_configurations
            ):
                if right_center_shared is not None:
                    physical = shared_configuration[right_center_shared]
                else:
                    physical = configuration[right_center_only]
                outgoing = layout.symmetry.fuse(
                    middle_charge,
                    layout.symmetry.physical_charges[physical],
                )
                for right_virtual, right_charge in enumerate(
                    layout.right_virtual_charges
                ):
                    if outgoing == right_charge:
                        columns.append(
                            configuration_flat * right_dim + right_virtual
                        )
            if not rows or not columns:
                continue
            block = matrix[np.ix_(rows, columns)]
            left_vectors, singular_values, right_vectors = np.linalg.svd(
                block, full_matrices=False
            )
            threshold = cutoff * singular_values[0] if singular_values.size else 0.0
            available = int(np.count_nonzero(singular_values > threshold))
            keep = min(len(middle_positions), available)
            total += float(np.sum(singular_values**2))
            discarded += float(np.sum(singular_values[keep:] ** 2))
            retained_in_shared += keep
            if keep == 0:
                continue
            bond_positions = np.asarray(middle_positions[:keep], dtype=int)
            left_factor = left_vectors[:, :keep]
            right_factor = right_vectors[:keep]
            retained_values = singular_values[:keep]
            if direction == "lr":
                right_factor = retained_values[:, None] * right_factor
            else:
                left_factor = left_factor * retained_values[None, :]
            left_matrix[np.ix_(rows, bond_positions)] = left_factor
            right_matrix[np.ix_(bond_positions, columns)] = right_factor

        sector_ranks.append(retained_in_shared)
        left_value = left_matrix.reshape(
            (left_dim,) + (physical_dim,) * left_count + (max_bond_dim,)
        )
        right_value = right_matrix.reshape(
            (max_bond_dim,) + (physical_dim,) * right_count + (right_dim,)
        )
        left_target = (
            (slice(None),)
            + (slice(None),) * left_count
            + shared_configuration
            + (slice(None),)
        )
        right_target = (
            (slice(None),)
            + shared_configuration
            + (slice(None),) * right_count
            + (slice(None),)
        )
        left_canonical[left_target] = left_value
        right_canonical[right_target] = right_value

    left_tensor = _tensor_to_neighborhood_order(
        left_canonical,
        layout.left_only + layout.shared,
        layout.left_neighborhood,
    )
    right_tensor = _tensor_to_neighborhood_order(
        right_canonical,
        layout.shared + layout.right_only,
        layout.right_neighborhood,
    )
    left_tensor[~layout.factor_mask("left")] = 0.0
    right_tensor[~layout.factor_mask("right")] = 0.0
    relative_discarded = discarded / total if total > 0.0 else 0.0
    return LETTASplit(
        left_tensor=left_tensor,
        right_tensor=right_tensor,
        discarded_weight=relative_discarded,
        sector_ranks=tuple(sector_ranks),
    )
