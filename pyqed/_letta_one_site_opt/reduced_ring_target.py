"""Covariant charge/spin closure for an actual reduced virtual ring.

An auxiliary dual-target leg couples the physical state to a scalar. It is a
representation label, not a physical orbital and not a determinant projection.
Norms trace that auxiliary leg; any scalar observable has the same normalized
expectation in each magnetic component of the physical target multiplet.
"""
from dataclasses import dataclass

import numpy as np

from pyqed.mps.su2 import SpinChargeSector
from pyqed.mps.symmetry import Sector
from .reduced_mpo_compile import SpinTensorMPO
from .reduced_symmetry import ReducedPhysicalBasis, _sector_irrep
from .reduced_ring_contraction import _validate_cycle, CyclicReducedNorm


def signed_sector(sector):
    """Represent a sector with signed U(1) labels, needed by a dual target.

    The older SpinChargeSector type deliberately permits particle counts only;
    the generic product Sector already supports negative dual charges.
    """
    if isinstance(sector, SpinChargeSector):
        return Sector(('charge', 'su2'), (sector.charge, sector.irrep))
    if isinstance(sector, Sector):
        return sector
    raise TypeError('ring sector must be a reduced product sector')


def signed_physical_basis(basis):
    return ReducedPhysicalBasis(basis.labels, tuple(map(signed_sector, basis.sectors)),
                                basis.multiplicities)


def dual_sector(sector):
    """Dualize additive charges; SU(2) and XOR point-group irreps are self-dual."""
    sector = signed_sector(sector)
    return Sector(sector.labels, tuple(
        value if label in {'su2', 'pg', 'point_group', 'abelianpg'} else -value
        for label, value in zip(sector.labels, sector.components)))


@dataclass(frozen=True)
class CyclicReducedMPO:
    """Native scalar MPO with site-specific physical and operator metadata."""
    sites: tuple
    physical_bases: tuple
    channels_by_site: tuple

    @classmethod
    def from_native(cls, mpo):
        if isinstance(mpo, cls):
            return mpo
        if not isinstance(mpo, SpinTensorMPO):
            raise TypeError('expected a native SpinTensorMPO')
        return cls(mpo.sites, (mpo.physical_basis,)*len(mpo.sites),
                   (mpo.channels,)*len(mpo.sites))

    def append_identity(self, physical_basis):
        """Extend H to H tensor I on a closure's dual-target representation."""
        identity = SpinTensorMPO.compile(
            (np.eye(physical_basis.dense_dim)[None, None],), physical_basis)
        return CyclicReducedMPO(self.sites + identity.sites,
            self.physical_bases + (physical_basis,),
            self.channels_by_site + (identity.channels,))


@dataclass(frozen=True)
class ReducedRingTarget:
    """Physical ring cores plus a covariant closure core.

    ``closure`` has axes (last virtual bond, dual target, first virtual bond).
    Its physical leg contains exactly one copy of the dual target irrep. Both
    adjacent virtual bonds remain unconstrained in dimension. The caller owns
    the closure multiplicity coefficients, just as it owns physical-site cores.
    Objects supplied here must remain immutable during a contraction.
    """
    physical_sites: tuple
    closure: object
    target: object

    def __post_init__(self):
        physical = tuple(self.physical_sites)
        if not physical:
            raise ValueError('a target ring requires physical sites')
        expected = dual_sector(self.target)
        if tuple(self.closure.qns[1]) != (expected,):
            raise ValueError('closure must carry exactly one copy of the dual target')
        _validate_cycle(physical + (self.closure,))
        object.__setattr__(self, 'physical_sites', physical)

    @property
    def sites(self):
        return self.physical_sites + (self.closure,)

    @property
    def target_basis(self):
        return ReducedPhysicalBasis(('dual-target',), (dual_sector(self.target),), (1,))

    @property
    def target_dimension(self):
        return _sector_irrep(self.target).dim

    def norm_squared(self):
        """Norm summed over auxiliary components (the scalar coupled state)."""
        return CyclicReducedNorm(self.sites).overlap()

    def component_norm_squared(self):
        """Norm of one unrescaled auxiliary slice, before multiplying by sqrt(2S+1)."""
        return self.norm_squared()/self.target_dimension

    def extend_hamiltonian(self, mpo):
        native = CyclicReducedMPO.from_native(mpo)
        if len(native.sites) != len(self.physical_sites):
            raise ValueError('Hamiltonian length must equal the number of physical sites')
        return native.append_identity(self.target_basis)
