"""Exact frontier-size contractions for reduced SU(2) LETTA tensors.

Only local Clebsch--Gordan component spaces are restored.  The many-body
wavefunction and Hamiltonian are never materialized. Cost is polynomial in the
explicit frontier-MPS dimensions; those dimensions can still grow
exponentially with graph frontier width for higher-dimensional orderings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyqed.mps.nonabelian.coupling import clebsch_gordan, ordered_two_m_values
from pyqed.mps.nonabelian.tensor import NonabelianTensor

from .reduced_symmetry import _sector_irrep


def _axis_sector_multiplicities(site, axis):
    dims = {}
    for key, block in site.data.items():
        sector = key[axis]
        dim = int(np.asarray(block).shape[axis])
        previous = dims.setdefault(sector, dim)
        if previous != dim:
            raise ValueError(
                f"inconsistent multiplicity for sector {sector!r} on axis {axis}"
            )
    sectors = tuple(
        sector for sector in dict.fromkeys(site.qns[axis]) if sector in dims
    )
    return sectors, dims


def _component_axis_layout(site, axis):
    sectors, multiplicities = _axis_sector_multiplicities(site, axis)
    offsets = {}
    cursor = 0
    for sector in sectors:
        size = multiplicities[sector] * _sector_irrep(sector).dim
        offsets[sector] = cursor
        cursor += size
    return offsets, multiplicities, cursor


def expand_reduced_mps_site(site):
    """Restore structural magnetic components of one reduced MPS tensor."""

    if not isinstance(site, NonabelianTensor) or site.rank != 3:
        raise TypeError("expand_reduced_mps_site expects a rank-3 NonabelianTensor")
    layouts = tuple(_component_axis_layout(site, axis) for axis in range(3))
    shape = tuple(layout[2] for layout in layouts)
    dtype = np.result_type(
        complex, *[np.asarray(block).dtype for block in site.data.values()]
    )
    dense = np.zeros(shape, dtype=dtype)
    for (q_left, q_phys, q_right), block in site.data.items():
        block = np.asarray(block)
        left_irrep = _sector_irrep(q_left)
        phys_irrep = _sector_irrep(q_phys)
        right_irrep = _sector_irrep(q_right)
        left_offset = layouts[0][0][q_left]
        phys_offset = layouts[1][0][q_phys]
        right_offset = layouts[2][0][q_right]
        left_ms = ordered_two_m_values(left_irrep)
        phys_ms = ordered_two_m_values(phys_irrep)
        right_ms = ordered_two_m_values(right_irrep)
        for left_copy in range(block.shape[0]):
            for phys_copy in range(block.shape[1]):
                for right_copy in range(block.shape[2]):
                    value = block[left_copy, phys_copy, right_copy]
                    if value == 0:
                        continue
                    for left_component, two_m_left in enumerate(left_ms):
                        for phys_component, two_m_phys in enumerate(phys_ms):
                            for right_component, two_m_right in enumerate(right_ms):
                                coefficient = clebsch_gordan(
                                    left_irrep,
                                    phys_irrep,
                                    right_irrep,
                                    two_m_left,
                                    two_m_phys,
                                    two_m_right,
                                )
                                if coefficient == 0:
                                    continue
                                dense[
                                    left_offset
                                    + left_copy * left_irrep.dim
                                    + left_component,
                                    phys_offset
                                    + phys_copy * phys_irrep.dim
                                    + phys_component,
                                    right_offset
                                    + right_copy * right_irrep.dim
                                    + right_component,
                                ] += value * coefficient
    if np.max(np.abs(dense.imag), initial=0.0) <= 1.0e-14:
        return dense.real
    return dense


def reduce_expanded_mps_site(site, dense):
    """Apply the exact adjoint of :func:`expand_reduced_mps_site`.

    ``site`` supplies the reduced block layout and fusion tree.  This is an
    adjoint projection, not a pseudoinverse: it is therefore the correct map
    for projected Hamiltonian and norm matvecs.
    """

    if not isinstance(site, NonabelianTensor) or site.rank != 3:
        raise TypeError("reduce_expanded_mps_site expects a rank-3 template")
    layouts = tuple(_component_axis_layout(site, axis) for axis in range(3))
    shape = tuple(layout[2] for layout in layouts)
    dense = np.asarray(dense)
    if dense.shape != shape:
        raise ValueError(
            f"expanded tensor has shape {dense.shape}, expected {shape}"
        )
    dtype = np.result_type(dense, complex)
    blocks = {
        key: np.zeros(np.asarray(block).shape, dtype=dtype)
        for key, block in site.data.items()
    }
    for (q_left, q_phys, q_right), block in blocks.items():
        left_irrep = _sector_irrep(q_left)
        phys_irrep = _sector_irrep(q_phys)
        right_irrep = _sector_irrep(q_right)
        left_offset = layouts[0][0][q_left]
        phys_offset = layouts[1][0][q_phys]
        right_offset = layouts[2][0][q_right]
        left_ms = ordered_two_m_values(left_irrep)
        phys_ms = ordered_two_m_values(phys_irrep)
        right_ms = ordered_two_m_values(right_irrep)
        for left_copy in range(block.shape[0]):
            for phys_copy in range(block.shape[1]):
                for right_copy in range(block.shape[2]):
                    value = 0.0j
                    for left_component, two_m_left in enumerate(left_ms):
                        for phys_component, two_m_phys in enumerate(phys_ms):
                            for right_component, two_m_right in enumerate(right_ms):
                                coefficient = clebsch_gordan(
                                    left_irrep,
                                    phys_irrep,
                                    right_irrep,
                                    two_m_left,
                                    two_m_phys,
                                    two_m_right,
                                )
                                if coefficient == 0:
                                    continue
                                value += np.conjugate(coefficient) * dense[
                                    left_offset
                                    + left_copy * left_irrep.dim
                                    + left_component,
                                    phys_offset
                                    + phys_copy * phys_irrep.dim
                                    + phys_component,
                                    right_offset
                                    + right_copy * right_irrep.dim
                                    + right_component,
                                ]
                    block[left_copy, phys_copy, right_copy] = value
    if all(
        np.max(np.abs(block.imag), initial=0.0) <= 1.0e-14
        for block in blocks.values()
    ):
        return {key: block.real for key, block in blocks.items()}
    return blocks


def _contract_left(environment, bra, core, ket):
    first = np.tensordot(environment, np.asarray(bra).conj(), axes=(1, 0))
    second = np.tensordot(first, core, axes=([0, 2], [0, 2]))
    updated = np.tensordot(second, ket, axes=([0, 3], [0, 1]))
    return updated.transpose(1, 0, 2)


def _contract_right(bra, core, environment, ket):
    first = np.tensordot(environment, np.asarray(bra).conj(), axes=(1, 2))
    second = np.tensordot(first, core, axes=([0, 3], [1, 2]))
    updated = np.tensordot(second, ket, axes=([0, 3], [2, 1]))
    return updated.transpose(1, 0, 2)


def _apply_one_site(environment_left, core, environment_right, tensor):
    return np.einsum(
        "xal,xyop,ybr,lpr->aob",
        environment_left,
        core,
        environment_right,
        tensor,
        optimize=True,
    )


def _apply_two_site(
    environment_left, core_left, core_right, environment_right, tensor
):
    return np.einsum(
        "xal,xyop,yzuv,zcs,lpvs->aouc",
        environment_left,
        core_left,
        core_right,
        environment_right,
        tensor,
        optimize=True,
    )


def _normalize_environment(environment, accumulated_log_scale):
    """Keep one boundary tensor in a safe range and track its scalar scale."""

    environment = np.asarray(environment)
    scale = float(np.max(np.abs(environment), initial=0.0))
    if scale == 0.0:
        return environment, float(accumulated_log_scale)
    return environment / scale, float(accumulated_log_scale + np.log(scale))


def _left_canonical_dense_sites(sites):
    """Return an equivalent, numerically conditioned dense MPS chain."""

    canonical = [np.asarray(site).copy() for site in sites]
    for site in range(len(canonical) - 1):
        left, physical, right = canonical[site].shape
        q_factor, r_factor = np.linalg.qr(
            canonical[site].reshape(left * physical, right), mode="reduced"
        )
        retained = q_factor.shape[1]
        canonical[site] = q_factor.reshape(left, physical, retained)
        canonical[site + 1] = np.tensordot(
            r_factor, canonical[site + 1], axes=(1, 0)
        )
    return tuple(canonical)


@dataclass(frozen=True)
class CanonicalEnvironmentChain:
    sites: tuple[np.ndarray, ...]
    cores: tuple[np.ndarray, ...]
    left: tuple[np.ndarray, ...]
    right: tuple[np.ndarray, ...]
    left_log_scales: tuple[float, ...]
    right_log_scales: tuple[float, ...]

    @classmethod
    def build(cls, sites, factors):
        dense_sites = tuple(expand_reduced_mps_site(site) for site in sites)
        cores = tuple(
            np.asarray(core.as_dense() if hasattr(core, "as_dense") else core)
            for core in factors
        )
        if len(dense_sites) != len(cores):
            raise ValueError("one canonical MPO factor is required per MPS site")
        for site, (tensor, core) in enumerate(zip(dense_sites, cores)):
            if core.ndim != 4 or core.shape[2:] != (tensor.shape[1], tensor.shape[1]):
                raise ValueError(
                    f"canonical MPO physical dimension at site {site} is incompatible"
                )

        initial_left = np.zeros(
            (cores[0].shape[0], dense_sites[0].shape[0], dense_sites[0].shape[0]),
            dtype=np.result_type(cores[0], dense_sites[0]),
        )
        initial_left[0] = np.eye(dense_sites[0].shape[0])
        left = [initial_left]
        left_log_scales = [0.0]
        for site in range(len(dense_sites) - 1):
            updated, log_scale = _normalize_environment(
                _contract_left(
                    left[-1], dense_sites[site], cores[site], dense_sites[site]
                ),
                left_log_scales[-1],
            )
            left.append(updated)
            left_log_scales.append(log_scale)

        initial_right = np.zeros(
            (
                cores[-1].shape[1],
                dense_sites[-1].shape[2],
                dense_sites[-1].shape[2],
            ),
            dtype=np.result_type(cores[-1], dense_sites[-1]),
        )
        initial_right[0] = np.eye(dense_sites[-1].shape[2])
        right = [initial_right]
        right_log_scales = [0.0]
        for site in range(len(dense_sites) - 1, 0, -1):
            updated, log_scale = _normalize_environment(
                _contract_right(
                    dense_sites[site], cores[site], right[-1], dense_sites[site]
                ),
                right_log_scales[-1],
            )
            right.append(updated)
            right_log_scales.append(log_scale)
        right.reverse()
        right_log_scales.reverse()
        return cls(
            dense_sites,
            cores,
            tuple(left),
            tuple(right),
            tuple(left_log_scales),
            tuple(right_log_scales),
        )

    def expectation(self):
        final = _contract_left(
            self.left[-1], self.sites[-1], self.cores[-1], self.sites[-1]
        )
        return np.trace(final[0]) * np.exp(self.left_log_scales[-1])

    def stable_expectation(self):
        """Recontract once in extended precision for final energy reporting."""

        stable_sites = _left_canonical_dense_sites(self.sites)
        is_complex = any(
            np.iscomplexobj(array) for array in stable_sites + self.cores
        )
        dtype = np.clongdouble if is_complex else np.longdouble
        environment = np.zeros(
            (
                self.cores[0].shape[0],
                stable_sites[0].shape[0],
                stable_sites[0].shape[0],
            ),
            dtype=dtype,
        )
        environment[0] = np.eye(stable_sites[0].shape[0], dtype=dtype)
        log_scale = np.longdouble(0.0)
        for bra, core in zip(stable_sites, self.cores):
            environment = _contract_left(
                environment,
                np.asarray(bra, dtype=dtype),
                np.asarray(core, dtype=dtype),
                np.asarray(bra, dtype=dtype),
            )
            scale = np.max(np.abs(environment), initial=np.longdouble(0.0))
            if scale != 0:
                environment /= scale
                log_scale += np.log(scale)
        return np.trace(environment[0]) * np.exp(log_scale)

    def local_matrix(self, site, source_frame):
        source_frame = np.asarray(source_frame)
        dense_shape = self.sites[int(site)].shape
        if source_frame.shape[0] != int(np.prod(dense_shape, dtype=int)):
            raise ValueError("source frame has incompatible expanded local dimension")
        batched = source_frame.reshape(dense_shape + (source_frame.shape[1],))
        applied = np.einsum(
            "xal,xyop,ybr,lprk->aobk",
            self.left[int(site)],
            self.cores[int(site)],
            self.right[int(site)],
            batched,
            optimize=True,
        ).reshape(source_frame.shape[0], source_frame.shape[1])
        applied *= np.exp(
            self.left_log_scales[int(site)]
            + self.right_log_scales[int(site)]
        )
        return source_frame.conj().T @ applied

    def local_action(self, site, tensor):
        """Apply one canonical effective operator without materializing it."""

        site = int(site)
        tensor = np.asarray(tensor)
        if tensor.shape != self.sites[site].shape:
            raise ValueError("local tensor has incompatible expanded shape")
        applied = _apply_one_site(
            self.left[site],
            self.cores[site],
            self.right[site],
            tensor,
        )
        return applied * np.exp(
            self.left_log_scales[site] + self.right_log_scales[site]
        )

    def pair_local_matrix(self, left_site, source_frame):
        left_site = int(left_site)
        source_frame = np.asarray(source_frame)
        dense_shape = (
            self.sites[left_site].shape[0],
            self.sites[left_site].shape[1],
            self.sites[left_site + 1].shape[1],
            self.sites[left_site + 1].shape[2],
        )
        if source_frame.shape[0] != int(np.prod(dense_shape, dtype=int)):
            raise ValueError("pair source frame has incompatible expanded dimension")
        batched = source_frame.reshape(dense_shape + (source_frame.shape[1],))
        applied = np.einsum(
            "xal,xyop,yzuv,zcs,lpvsk->aouck",
            self.left[left_site],
            self.cores[left_site],
            self.cores[left_site + 1],
            self.right[left_site + 1],
            batched,
            optimize=True,
        ).reshape(source_frame.shape[0], source_frame.shape[1])
        applied *= np.exp(
            self.left_log_scales[left_site]
            + self.right_log_scales[left_site + 1]
        )
        return source_frame.conj().T @ applied

    def pair_action(self, left_site, tensor):
        """Apply one adjacent-pair effective operator matrix-free."""

        left_site = int(left_site)
        tensor = np.asarray(tensor)
        expected = (
            self.sites[left_site].shape[0],
            self.sites[left_site].shape[1],
            self.sites[left_site + 1].shape[1],
            self.sites[left_site + 1].shape[2],
        )
        if tensor.shape != expected:
            raise ValueError("pair tensor has incompatible expanded shape")
        applied = _apply_two_site(
            self.left[left_site],
            self.cores[left_site],
            self.cores[left_site + 1],
            self.right[left_site + 1],
            tensor,
        )
        return applied * np.exp(
            self.left_log_scales[left_site]
            + self.right_log_scales[left_site + 1]
        )


def identity_canonical_factors(sites, *, dtype=complex):
    factors = []
    for site in sites:
        dimension = expand_reduced_mps_site(site).shape[1]
        factors.append(np.eye(dimension, dtype=dtype)[None, None, :, :])
    return tuple(factors)


__all__ = [
    "CanonicalEnvironmentChain",
    "expand_reduced_mps_site",
    "identity_canonical_factors",
    "reduce_expanded_mps_site",
]
