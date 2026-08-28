"""Model-independent reduced physical legs and common SU(2) operators."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np

from pyqed.mps.nonabelian.builder import AutoMPO
from pyqed.mps.nonabelian.mpo import PhysicalLeg
from pyqed.mps.nonabelian.operators import ReducedTensorOperator
from pyqed.mps.su2 import SU2Irrep

from .reduced_symmetry import ReducedPhysicalBasis, _sector_irrep


@dataclass(frozen=True)
class ReducedMPOHamiltonian:
    """Compact reduced MPO plus an exact local magnetic-component view."""

    factors: tuple
    canonical_factors: tuple
    name: str = "reduced MPO"

    def __post_init__(self):
        factors = tuple(self.factors)
        canonical = tuple(self.canonical_factors)
        if not factors or len(factors) != len(canonical):
            raise ValueError(
                "factors and canonical_factors must be nonempty equal-length chains"
            )
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "canonical_factors", canonical)

    def __len__(self):
        return len(self.factors)

    def __iter__(self):
        return iter(self.factors)

    def __getitem__(self, item):
        return self.factors[item]


def physical_leg_from_reduced_basis(physical_basis, *, fully_reduced=True):
    """Return non-Abelian MPO metadata for a user-defined reduced local basis.

    In the fully reduced form each physical sector dimension is its outer
    multiplicity.  Setting ``fully_reduced=False`` restores magnetic irrep
    components and is useful for dense reference checks.
    """

    if not isinstance(physical_basis, ReducedPhysicalBasis):
        raise TypeError("physical_basis must be ReducedPhysicalBasis")
    dims = {}
    for sector, multiplicity in zip(
        physical_basis.sectors, physical_basis.multiplicities
    ):
        dims[sector] = int(multiplicity) * (
            1 if fully_reduced else _sector_irrep(sector).dim
        )
    return PhysicalLeg.from_dims(dims, sectors=physical_basis.sectors)


def su2_spin_operator(
    physical_leg,
    *,
    physical_basis,
    fully_reduced,
):
    r"""Construct the rank-one angular-momentum tensor ``J``.

    The reduced matrix element in a spin-``j`` irrep is
    ``sqrt(j (j + 1) (2j + 1))``.  This convention expands to
    ``(-J_+ / sqrt(2), J_z, J_- / sqrt(2))`` for spherical components
    ``(+1, 0, -1)``.
    """

    if not isinstance(physical_leg, PhysicalLeg):
        raise TypeError("physical_leg must be PhysicalLeg")
    if not isinstance(physical_basis, ReducedPhysicalBasis):
        raise TypeError("physical_basis must be ReducedPhysicalBasis")
    if not isinstance(fully_reduced, (bool, np.bool_)):
        raise TypeError("fully_reduced must be an explicit boolean")
    expected_leg = physical_leg_from_reduced_basis(
        physical_basis, fully_reduced=bool(fully_reduced)
    )
    if physical_leg != expected_leg:
        representation = "fully reduced" if fully_reduced else "canonical"
        raise ValueError(
            f"physical_leg does not match the {representation} representation "
            "of physical_basis"
        )
    if any(multiplicity != 1 for multiplicity in physical_basis.multiplicities):
        raise NotImplementedError(
            "su2_spin_operator does not infer an action in outer-multiplicity "
            "spaces; supply a custom ReducedTensorOperator explicitly"
        )
    blocks = {}
    for sector in physical_basis.sectors:
        irrep = _sector_irrep(sector)
        if irrep.two_j == 0:
            continue
        j = irrep.j
        blocks[(sector, sector)] = np.sqrt(j * (j + 1.0) * (2.0 * j + 1.0))
    if not blocks:
        raise ValueError("the physical leg contains no non-scalar SU(2) irrep")
    return ReducedTensorOperator(
        reduced_blocks=blocks,
        phys_out_leg=physical_leg,
        phys_in_leg=physical_leg,
        rank_irrep=SU2Irrep(2),
    )


def su2_heisenberg_mpo(
    nsites,
    *,
    physical_basis=None,
    physical_leg=None,
    coupling=1.0,
    periodic=False,
):
    r"""Build ``coupling * sum_<ij> J_i dot J_j`` as a reduced SU(2) MPO.

    The compact factors always match the fully reduced ``physical_basis`` and
    can be passed directly to reduced LETTA optimization. ``physical_leg`` may
    override the full magnetic-component leg used by the exact canonical
    contraction view; when omitted it is inferred from ``physical_basis``.
    """

    try:
        nsites = index(nsites)
    except TypeError as error:
        raise ValueError("nsites must be an integer") from error
    if nsites < 2:
        raise ValueError("the Heisenberg chain needs at least two sites")
    if physical_basis is None:
        physical_basis = ReducedPhysicalBasis.spin_half()
    if not isinstance(physical_basis, ReducedPhysicalBasis):
        raise TypeError("physical_basis must be ReducedPhysicalBasis")
    reduced_leg = physical_leg_from_reduced_basis(physical_basis)
    expected_canonical_leg = physical_leg_from_reduced_basis(
        physical_basis, fully_reduced=False
    )
    canonical_leg = expected_canonical_leg if physical_leg is None else physical_leg
    if canonical_leg != expected_canonical_leg:
        raise ValueError(
            "physical_leg must be the full magnetic-component leg implied by "
            "physical_basis"
        )

    def build_for_leg(leg, *, fully_reduced):
        spin = su2_spin_operator(
            leg,
            physical_basis=physical_basis,
            fully_reduced=fully_reduced,
        )
        builder = AutoMPO([leg] * nsites)
        # AutoMPO's doubled-component reduced-string convention evaluates the
        # spin-half scalar coupling as -2 J.J/sqrt(3).
        scalar_coefficient = -0.5 * np.sqrt(3.0) * np.asarray(coupling)
        for site in range(nsites - 1):
            builder.add_reduced_string(
                (site, spin),
                (site + 1, spin),
                intermediate_irreps=(SU2Irrep(2),),
                coeff=scalar_coefficient,
                family="heisenberg",
            )
        if periodic and nsites > 2:
            builder.add_reduced_string(
                (0, spin),
                (nsites - 1, spin),
                intermediate_irreps=(SU2Irrep(2),),
                coeff=scalar_coefficient,
                family="heisenberg_periodic",
            )
        return tuple(builder.build())

    factors = build_for_leg(reduced_leg, fully_reduced=True)
    canonical_factors = build_for_leg(canonical_leg, fully_reduced=False)
    return ReducedMPOHamiltonian(
        factors=factors,
        canonical_factors=canonical_factors,
        name="SU(2) Heisenberg",
    )


__all__ = [
    "ReducedMPOHamiltonian",
    "physical_leg_from_reduced_basis",
    "su2_heisenberg_mpo",
    "su2_spin_operator",
]
