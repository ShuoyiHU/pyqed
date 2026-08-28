import pytest

from pyqed._letta_one_site_opt import (
    ReducedPhysicalBasis,
    ReducedSymmetry,
)
from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
from pyqed.mps.symmetry import AbelianSector, Sector


def test_spin_half_basis_stores_one_reduced_state_for_two_dense_components():
    basis = ReducedPhysicalBasis.spin_half()

    assert basis.labels == ("spin-half",)
    assert basis.sectors == (SpinChargeSector(0, SU2Irrep(1)),)
    assert basis.multiplicities == (1,)
    assert basis.reduced_dim == 1
    assert basis.dense_dim == 2
    assert basis.reduced_states[0].sector == basis.sectors[0]
    assert basis.reduced_states[0].copy == 0


def test_user_defined_basis_supports_irrep_multiplicity_spaces():
    singlet = SpinChargeSector(0, SU2Irrep(0))
    doublet = SpinChargeSector(1, SU2Irrep(1))
    basis = ReducedPhysicalBasis(
        labels=("vacuum", "flavor"),
        sectors=(singlet, doublet),
        multiplicities=(1, 3),
    )

    assert basis.reduced_dim == 4
    assert basis.dense_dim == 7
    assert [state.copy for state in basis.reduced_states_for_sector(doublet)] == [0, 1, 2]


def test_spatial_orbital_basis_matches_existing_fully_reduced_site():
    basis = ReducedPhysicalBasis.spatial_orbital()

    assert basis.labels == ("empty", "single", "double")
    assert tuple(sector.charge for sector in basis.sectors) == (0, 1, 2)
    assert tuple(sector.irrep.two_j for sector in basis.sectors) == (0, 1, 0)
    assert basis.multiplicities == (1, 1, 1)
    assert basis.reduced_dim == 3
    assert basis.dense_dim == 4


def test_reduced_symmetry_computes_target_compatible_spin_half_bonds():
    symmetry = ReducedSymmetry.su2(
        ReducedPhysicalBasis.spin_half(),
        target_two_j=0,
    )

    bonds = symmetry.reachable_bond_sectors(4)

    assert bonds[0] == (SpinChargeSector(0, SU2Irrep(0)),)
    assert bonds[1] == (SpinChargeSector(0, SU2Irrep(1)),)
    assert bonds[2] == (
        SpinChargeSector(0, SU2Irrep(0)),
        SpinChargeSector(0, SU2Irrep(2)),
    )
    assert bonds[3] == (SpinChargeSector(0, SU2Irrep(1)),)
    assert bonds[4] == (SpinChargeSector(0, SU2Irrep(0)),)


def test_reduced_symmetry_rejects_unreachable_target_sector():
    symmetry = ReducedSymmetry.su2(
        ReducedPhysicalBasis.spin_half(),
        target_two_j=0,
    )

    with pytest.raises(ValueError, match="not reachable"):
        symmetry.reachable_bond_sectors(3)


def test_conditioning_labels_are_reduced_multiplicity_labels_not_m_states():
    basis = ReducedPhysicalBasis.spin_half()

    assert basis.condition_labels == (basis.reduced_states[0],)
    with pytest.raises(ValueError, match="magnetic-component"):
        basis.condition_index("up")


def test_reduced_symmetry_allocates_whole_multiplets_on_every_bond():
    symmetry = ReducedSymmetry.su2(
        ReducedPhysicalBasis.spin_half(),
        target_two_j=0,
    )

    allocation = symmetry.allocate_bond_sectors(4, multiplets_per_sector=2)

    assert allocation[0] == (
        SpinChargeSector(0, SU2Irrep(1)),
        SpinChargeSector(0, SU2Irrep(1)),
    )
    assert allocation[1] == (
        SpinChargeSector(0, SU2Irrep(0)),
        SpinChargeSector(0, SU2Irrep(0)),
        SpinChargeSector(0, SU2Irrep(2)),
        SpinChargeSector(0, SU2Irrep(2)),
    )
    assert allocation[2] == allocation[0]


def test_reduced_basis_rejects_component_count_as_multiplicity():
    with pytest.raises(ValueError, match="multiplicity"):
        ReducedPhysicalBasis(
            labels=("up", "down"),
            sectors=(
                SpinChargeSector(0, SU2Irrep(1)),
                SpinChargeSector(0, SU2Irrep(1)),
            ),
            multiplicities=(1, 1),
        )


def test_generic_product_sector_supports_user_defined_charge_times_su2():
    identity = Sector(("charge", "su2"), (0, SU2Irrep(0)))
    physical = Sector(("charge", "su2"), (1, SU2Irrep(1)))
    target = Sector(("charge", "su2"), (2, SU2Irrep(0)))
    basis = ReducedPhysicalBasis(
        labels=("custom-doublet",),
        sectors=(physical,),
        multiplicities=(1,),
    )
    symmetry = ReducedSymmetry(
        physical_basis=basis,
        identity=identity,
        sector=target,
        name="custom charge x SU(2)",
    )

    assert symmetry.reachable_bond_sectors(2)[-1] == (target,)
    assert identity.is_abelian is False
    assert Sector(("charge",), (0,)).is_abelian is True
    assert AbelianSector(("charge",), (0,)).is_abelian is True
