"""Exact factor-graph frontier embedding for reduced conditional LETTA states."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from pyqed.mps.nonabelian.coupling import clebsch_gordan, ordered_two_m_values
from pyqed.mps.nonabelian.tensor import NonabelianTensor

from .reduced_state import ReducedLatticeLETTA
from .reduced_symmetry import ReducedPhysicalBasis, _sector_irrep


class _BlockVectorLayout:
    def __init__(self, shapes):
        self.keys = tuple(sorted(shapes))
        self.shapes = {key: tuple(int(value) for value in shapes[key]) for key in self.keys}
        offsets = {}
        cursor = 0
        for key in self.keys:
            size = int(np.prod(self.shapes[key], dtype=int))
            offsets[key] = (cursor, cursor + size)
            cursor += size
        self.offsets = offsets
        self.size = cursor

    def pack(self, blocks):
        dtype = np.result_type(
            *[np.asarray(blocks[key]).dtype for key in self.keys],
            float,
        )
        vector = np.zeros(self.size, dtype=dtype)
        for key in self.keys:
            block = np.asarray(blocks[key])
            if block.shape != self.shapes[key]:
                raise ValueError(
                    f"block {key!r} has shape {block.shape}, expected {self.shapes[key]}"
                )
            start, stop = self.offsets[key]
            vector[start:stop] = block.reshape(-1)
        return vector

    def unpack(self, vector):
        vector = np.asarray(vector).reshape(-1)
        if vector.size != self.size:
            raise ValueError(f"packed vector has size {vector.size}, expected {self.size}")
        return {
            key: vector[start:stop].reshape(self.shapes[key]).copy()
            for key, (start, stop) in self.offsets.items()
        }


@dataclass(frozen=True)
class FrontierSiteEmbedding:
    """Sparse linear map from one reduced LETTA core to a frontier MPS site."""

    source_layout: _BlockVectorLayout
    target_layout: _BlockVectorLayout
    source_indices: np.ndarray
    target_indices: np.ndarray
    left_variables: tuple[int, ...]
    right_variables: tuple[int, ...]

    @property
    def source_size(self):
        return self.source_layout.size

    @property
    def target_size(self):
        return self.target_layout.size

    def apply(self, source):
        source = np.asarray(source).reshape(-1)
        if source.size != self.source_size:
            raise ValueError("source vector has incompatible size")
        target = np.zeros(self.target_size, dtype=source.dtype)
        target[self.target_indices] = source[self.source_indices]
        return target

    def adjoint(self, target):
        target = np.asarray(target).reshape(-1)
        if target.size != self.target_size:
            raise ValueError("target vector has incompatible size")
        source = np.zeros(self.source_size, dtype=target.dtype)
        np.add.at(source, self.source_indices, target[self.target_indices])
        return source

    def pack_source(self, blocks):
        return self.source_layout.pack(blocks)

    def unpack_source(self, vector):
        return self.source_layout.unpack(vector)

    def pack_target(self, blocks):
        return self.target_layout.pack(blocks)

    def unpack_target(self, vector):
        return self.target_layout.unpack(vector)

    def expand_blocks(self, blocks):
        return self.unpack_target(self.apply(self.pack_source(blocks)))

    def dense_matrix(self):
        matrix = np.zeros((self.target_size, self.source_size), dtype=float)
        matrix[self.target_indices, self.source_indices] = 1.0
        return matrix


class ReducedFrontier:
    """Physical-variable frontier for exact sequential LETTA contraction."""

    def __init__(self, neighborhoods, reduced_dim):
        self.neighborhoods = tuple(tuple(int(value) for value in item) for item in neighborhoods)
        self.nsites = len(self.neighborhoods)
        self.reduced_dim = int(reduced_dim)
        if self.reduced_dim <= 0:
            raise ValueError("reduced_dim must be positive")
        occurrences = [set([site]) for site in range(self.nsites)]
        for factor_site, neighborhood in enumerate(self.neighborhoods):
            for variable in neighborhood:
                if variable < 0 or variable >= self.nsites:
                    raise ValueError("frontier neighborhood variable out of range")
                occurrences[variable].add(factor_site)
        intervals = tuple((min(items), max(items)) for items in occurrences)
        self.intervals = intervals
        self.cuts = tuple(
            tuple(
                variable
                for variable, (first, last) in enumerate(intervals)
                if first <= cut < last
            )
            for cut in range(self.nsites - 1)
        )

    @classmethod
    def from_state(cls, state):
        if not isinstance(state, ReducedLatticeLETTA):
            raise TypeError("ReducedFrontier.from_state expects ReducedLatticeLETTA")
        return cls(state._neighborhoods, state.physical_dim)

    def left_variables(self, site):
        return () if int(site) == 0 else self.cuts[int(site) - 1]

    def right_variables(self, site):
        return () if int(site) == self.nsites - 1 else self.cuts[int(site)]

    def _assignments(self, variables):
        if not variables:
            return ((),)
        return tuple(product(range(self.reduced_dim), repeat=len(variables)))

    def site_embedding(self, state, site):
        if not isinstance(state, ReducedLatticeLETTA):
            raise TypeError("site_embedding expects ReducedLatticeLETTA")
        site = int(site)
        if not 0 <= site < self.nsites:
            raise IndexError("site index out of range")
        if state.nsites != self.nsites or state.physical_dim != self.reduced_dim:
            raise ValueError("frontier and reduced LETTA state are incompatible")

        left_variables = self.left_variables(site)
        right_variables = self.right_variables(site)
        left_assignments = self._assignments(left_variables)
        right_assignments = self._assignments(right_variables)
        source_shapes = {key: block.shape for key, block in state.tensors[site].items()}
        target_shapes = {
            key: (
                block.shape[0] * len(left_assignments),
                block.shape[1],
                block.shape[-1] * len(right_assignments),
            )
            for key, block in state.tensors[site].items()
        }
        source_layout = _BlockVectorLayout(source_shapes)
        target_layout = _BlockVectorLayout(target_shapes)
        state_index = {
            (item.sector, item.copy): idx
            for idx, item in enumerate(state.physical_basis.reduced_states)
        }

        source_indices = []
        target_indices = []
        neighborhood = state.site_neighborhood(site)
        for key in source_layout.keys:
            q_left, q_phys, _q_right = key
            del q_left
            source_shape = source_layout.shapes[key]
            target_shape = target_layout.shapes[key]
            source_offset = source_layout.offsets[key][0]
            target_offset = target_layout.offsets[key][0]
            d_left = source_shape[0]
            d_phys = source_shape[1]
            d_right = source_shape[-1]
            for left_memory, left_values in enumerate(left_assignments):
                left_map = dict(zip(left_variables, left_values))
                for physical_copy in range(d_phys):
                    physical_value = state_index[(q_phys, physical_copy)]
                    for right_memory, right_values in enumerate(right_assignments):
                        assignment = dict(left_map)
                        consistent = True
                        if site in assignment and assignment[site] != physical_value:
                            consistent = False
                        assignment[site] = physical_value
                        for variable, value in zip(right_variables, right_values):
                            if variable in assignment and assignment[variable] != value:
                                consistent = False
                                break
                            assignment[variable] = value
                        if not consistent or any(
                            variable not in assignment for variable in neighborhood
                        ):
                            continue
                        dependencies = tuple(
                            assignment[variable] for variable in neighborhood[1:]
                        )
                        for left_slot in range(d_left):
                            for right_slot in range(d_right):
                                source_local = np.ravel_multi_index(
                                    (left_slot, physical_copy)
                                    + dependencies
                                    + (right_slot,),
                                    source_shape,
                                )
                                target_local = np.ravel_multi_index(
                                    (
                                        left_memory * d_left + left_slot,
                                        physical_copy,
                                        right_memory * d_right + right_slot,
                                    ),
                                    target_shape,
                                )
                                source_indices.append(source_offset + source_local)
                                target_indices.append(target_offset + target_local)
        target_indices = np.asarray(target_indices, dtype=int)
        if target_indices.size != np.unique(target_indices).size:
            raise RuntimeError("frontier embedding assigned one MPS entry more than once")
        return FrontierSiteEmbedding(
            source_layout=source_layout,
            target_layout=target_layout,
            source_indices=np.asarray(source_indices, dtype=int),
            target_indices=target_indices,
            left_variables=left_variables,
            right_variables=right_variables,
        )

    def to_mps(self, state):
        sites = []
        for site in range(self.nsites):
            embedding = self.site_embedding(state, site)
            blocks = embedding.expand_blocks(state.tensors[site])
            left_memory = self.reduced_dim ** len(self.left_variables(site))
            right_memory = self.reduced_dim ** len(self.right_variables(site))
            left_qns = [
                sector
                for _memory in range(left_memory)
                for sector in state.left_virtual_sectors(site)
            ]
            right_qns = [
                sector
                for _memory in range(right_memory)
                for sector in state.right_virtual_sectors(site)
            ]
            physical_qns = [
                sector
                for sector, multiplicity in zip(
                    state.physical_basis.sectors,
                    state.physical_basis.multiplicities,
                )
                for _ in range(multiplicity)
            ]
            sites.append(
                NonabelianTensor(
                    data=blocks,
                    qns=[left_qns, physical_qns, right_qns],
                    dirs=[-1, 1, 1],
                    metadata={
                        "physical_basis": "fully_reduced_su2",
                        "letta_frontier_left": self.left_variables(site),
                        "letta_frontier_right": self.right_variables(site),
                    },
                )
            )
        return sites


def reduced_mps_state_vector(
    sites,
    physical_basis,
    *,
    target_sector,
    target_two_m=None,
    max_sites=14,
):
    """Expand a fully reduced MPS into one target multiplet component."""

    sites = tuple(sites)
    if not sites:
        raise ValueError("reduced MPS needs at least one site")
    if len(sites) > int(max_sites):
        raise ValueError("dense reduced-MPS reconstruction is verification-only")
    if not isinstance(physical_basis, ReducedPhysicalBasis):
        raise TypeError("physical_basis must be ReducedPhysicalBasis")
    target_irrep = _sector_irrep(target_sector)
    if target_two_m is None:
        target_two_m = target_irrep.two_j
    if int(target_two_m) not in ordered_two_m_values(target_irrep):
        raise ValueError("target_two_m is not a component of target_sector")
    dense_states = tuple(
        (state, two_m)
        for state in physical_basis.reduced_states
        for two_m in ordered_two_m_values(state.irrep)
    )
    vector = np.zeros(
        len(dense_states) ** len(sites),
        dtype=np.result_type(
            complex,
            *[block.dtype for tensor in sites for block in tensor.data.values()],
        ),
    )
    dimensions = (len(dense_states),) * len(sites)
    identity = sites[0].qns[0][0]
    for flat, configuration in enumerate(np.ndindex(*dimensions)):
        boundary = {(identity, 0, 0): 1.0 + 0.0j}
        for tensor, physical_index in zip(sites, configuration):
            physical_state, two_m_phys = dense_states[physical_index]
            updated = {}
            for (q_left, left_slot, two_m_left), amplitude in boundary.items():
                for (block_left, block_phys, block_right), block in tensor.data.items():
                    if block_left != q_left or block_phys != physical_state.sector:
                        continue
                    arr = np.asarray(block)
                    for right_slot in range(arr.shape[2]):
                        reduced_value = arr[left_slot, physical_state.copy, right_slot]
                        if reduced_value == 0:
                            continue
                        for two_m_right in ordered_two_m_values(_sector_irrep(block_right)):
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
        vector[flat] = boundary.get((target_sector, 0, int(target_two_m)), 0.0)
    if np.max(np.abs(vector.imag), initial=0.0) <= 1.0e-13:
        return vector.real
    return vector


__all__ = [
    "FrontierSiteEmbedding",
    "ReducedFrontier",
    "reduced_mps_state_vector",
]
