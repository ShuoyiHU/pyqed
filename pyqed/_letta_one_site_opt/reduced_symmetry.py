"""User-defined reduced local bases and exact SU(2) fusion data for LETTA."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

from pyqed.mps.su2 import (
    SpinChargeSector,
    SpatialOrbitalSite,
    SU2Irrep,
    fuse_charge_spin_sectors,
)
from pyqed.mps.symmetry import Sector


def _sector_irrep(sector):
    if isinstance(sector, SpinChargeSector):
        return sector.irrep
    if isinstance(sector, Sector) and "su2" in sector.labels:
        irrep = sector.components[sector.labels.index("su2")]
        if isinstance(irrep, SU2Irrep):
            return irrep
    raise TypeError(
        "reduced physical sectors must contain an SU(2) irrep; expected "
        "SpinChargeSector or Sector(..., 'su2', ...)"
    )


def _fuse_sectors(left, right):
    if isinstance(left, SpinChargeSector) and isinstance(right, SpinChargeSector):
        return tuple(fuse_charge_spin_sectors(left, right))
    if isinstance(left, Sector) and isinstance(right, Sector):
        fused = left.fuse(right)
        if fused is NotImplemented:
            raise TypeError("incompatible product-sector labels")
        return tuple(fused)
    raise TypeError(
        f"cannot fuse reduced sectors {type(left).__name__} and {type(right).__name__}"
    )


@dataclass(frozen=True, order=True)
class ReducedBasisState:
    """One scalar multiplicity coordinate associated with a complete irrep."""

    label: str
    sector: object
    copy: int = 0

    @property
    def irrep(self):
        return _sector_irrep(self.sector)


@dataclass(frozen=True)
class ReducedPhysicalBasis:
    r"""Local decomposition ``direct_sum_a V[j_a] tensor C[m_a]``.

    ``multiplicities`` count copies of complete irreducible representations.
    Magnetic components are structural Clebsch-Gordan coordinates and are not
    independent LETTA conditioning labels.
    """

    labels: tuple[str, ...]
    sectors: tuple[object, ...]
    multiplicities: tuple[int, ...]

    def __post_init__(self):
        labels = tuple(str(label) for label in self.labels)
        sectors = tuple(self.sectors)
        try:
            multiplicities = tuple(index(value) for value in self.multiplicities)
        except TypeError as error:
            raise ValueError("physical irrep multiplicities must be integers") from error
        if not labels:
            raise ValueError("a reduced physical basis needs at least one irrep sector")
        if not (len(labels) == len(sectors) == len(multiplicities)):
            raise ValueError("labels, sectors, and multiplicities must have equal lengths")
        if len(set(labels)) != len(labels):
            raise ValueError("reduced physical labels must be unique")
        if len(set(sectors)) != len(sectors):
            raise ValueError(
                "repeat copies through the multiplicity field, not duplicate irrep sectors; "
                "magnetic components are not multiplicity copies"
            )
        if any(value <= 0 for value in multiplicities):
            raise ValueError("physical irrep multiplicity must be positive")
        for sector in sectors:
            _sector_irrep(sector)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "sectors", sectors)
        object.__setattr__(self, "multiplicities", multiplicities)

    @classmethod
    def spin_half(cls, *, charge=0, label="spin-half"):
        return cls(
            labels=(label,),
            sectors=(SpinChargeSector(int(charge), SU2Irrep(1)),),
            multiplicities=(1,),
        )

    @classmethod
    def spatial_orbital(cls):
        site = SpatialOrbitalSite()
        return cls(
            labels=("empty", "single", "double"),
            sectors=tuple(site.qn),
            multiplicities=(1, 1, 1),
        )

    @property
    def reduced_states(self):
        return tuple(
            ReducedBasisState(label, sector, copy)
            for label, sector, multiplicity in zip(
                self.labels, self.sectors, self.multiplicities
            )
            for copy in range(multiplicity)
        )

    @property
    def condition_labels(self):
        """Scalar labels permitted on repeated LETTA dependency axes."""

        return self.reduced_states

    @property
    def reduced_dim(self):
        return sum(self.multiplicities)

    @property
    def dense_dim(self):
        return sum(
            multiplicity * _sector_irrep(sector).dim
            for sector, multiplicity in zip(self.sectors, self.multiplicities)
        )

    def reduced_states_for_sector(self, sector):
        return tuple(state for state in self.reduced_states if state.sector == sector)

    def condition_index(self, label):
        states = self.reduced_states
        if isinstance(label, ReducedBasisState):
            try:
                return states.index(label)
            except ValueError as error:
                raise ValueError("reduced basis state does not belong to this basis") from error
        if isinstance(label, int):
            if 0 <= label < len(states):
                return label
            raise IndexError("reduced conditioning index out of range")
        matches = [
            idx for idx, state in enumerate(states) if state.label == str(label)
        ]
        if len(matches) == 1:
            return matches[0]
        if str(label).lower() in {
            "up",
            "down",
            "m=+1/2",
            "m=-1/2",
            "+1/2",
            "-1/2",
        }:
            raise ValueError(
                "magnetic-component labels cannot be used as exact SU(2) LETTA "
                "conditioning labels; use one label for the complete multiplet"
            )
        if len(matches) > 1:
            raise ValueError("conditioning label is ambiguous; provide ReducedBasisState")
        raise ValueError(f"unknown reduced conditioning label {label!r}")


@dataclass(frozen=True)
class ReducedSymmetry:
    """Exact reduced SU(2), optionally combined with additive charge."""

    physical_basis: ReducedPhysicalBasis
    identity: object
    sector: object
    name: str = "SU(2)"

    def __post_init__(self):
        if not isinstance(self.physical_basis, ReducedPhysicalBasis):
            raise TypeError("physical_basis must be a ReducedPhysicalBasis")
        _sector_irrep(self.identity)
        _sector_irrep(self.sector)
        if type(self.identity) is not type(self.sector):
            raise TypeError("identity and target sectors must use the same sector type")
        for physical in self.physical_basis.sectors:
            if type(physical) is not type(self.identity):
                raise TypeError("all physical, identity, and target sectors must share a type")

    @classmethod
    def su2(
        cls,
        physical_basis,
        *,
        target_two_j,
        target_charge=0,
        name="SU(2)",
    ):
        return cls(
            physical_basis=physical_basis,
            identity=SpinChargeSector(0, SU2Irrep(0)),
            sector=SpinChargeSector(int(target_charge), SU2Irrep(int(target_two_j))),
            name=name,
        )

    def fuse(self, left, physical):
        return _fuse_sectors(left, physical)

    def reachable_bond_sectors(self, nsites):
        try:
            nsites = index(nsites)
        except TypeError as error:
            raise ValueError("nsites must be an integer") from error
        if nsites <= 0:
            raise ValueError("nsites must be positive")

        forward = [{self.identity}]
        for _ in range(nsites):
            next_sectors = set()
            for left in forward[-1]:
                for physical in self.physical_basis.sectors:
                    next_sectors.update(self.fuse(left, physical))
            forward.append(next_sectors)
        if self.sector not in forward[-1]:
            raise ValueError(
                f"target sector {self.sector!r} is not reachable with {nsites} sites"
            )

        compatible = [set() for _ in range(nsites + 1)]
        compatible[-1].add(self.sector)
        for site in range(nsites - 1, -1, -1):
            for left in forward[site]:
                if any(
                    right in compatible[site + 1]
                    for physical in self.physical_basis.sectors
                    for right in self.fuse(left, physical)
                ):
                    compatible[site].add(left)
        if self.identity not in compatible[0]:
            raise ValueError(
                f"target sector {self.sector!r} is not reachable from the identity"
            )
        return tuple(tuple(sorted(sectors)) for sectors in compatible)

    def allocate_bond_sectors(self, nsites, *, multiplets_per_sector=1):
        try:
            copies = index(multiplets_per_sector)
        except TypeError as error:
            raise ValueError("multiplets_per_sector must be an integer") from error
        if copies <= 0:
            raise ValueError("multiplets_per_sector must be positive")
        reachable = self.reachable_bond_sectors(nsites)
        return tuple(
            tuple(sector for sector in bond for _ in range(copies))
            for bond in reachable[1:-1]
        )


__all__ = [
    "ReducedBasisState",
    "ReducedPhysicalBasis",
    "ReducedSymmetry",
]
