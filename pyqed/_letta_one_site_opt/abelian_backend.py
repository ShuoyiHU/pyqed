"""Lossless U(1) coordinate adapter for the shared reduced LETTA solvers.

Abelian irreps have dimension one. Appending a trivial spin representation lets
charge sectors use the same native block contractions and factor compressors;
no wavefunction projection, charge penalty, or determinant expansion is needed.
"""
from dataclasses import replace

import numpy as np

from pyqed.mps.symmetry import Sector
from pyqed.mps.su2 import SU2Irrep
from .symmetry import AbelianSymmetry
from .state import LatticeLETTA
from .operators import LatticeMPO
from .reduced_state import ReducedLatticeLETTA
from .reduced_symmetry import ReducedPhysicalBasis, ReducedSymmetry
from .reduced_operators import ReducedMPOHamiltonian


class AbelianReducedMap:
    """Group equal physical charges and keep the original basis permutation."""

    def __init__(self, symmetry):
        if not isinstance(symmetry, AbelianSymmetry):
            raise TypeError('expected AbelianSymmetry')
        if any(m is not None for m in symmetry._moduli):
            raise ValueError('the reduced Abelian adapter currently supports U(1) products only')
        self.symmetry = symmetry
        self.labels = tuple(f'u1_{i}' for i in range(symmetry.number_of_components))+('su2',)
        self.physical_groups = {}
        for i, charge in enumerate(symmetry.physical_charges):
            self.physical_groups.setdefault(self.to_sector(charge), []).append(i)
        self.physical_order = tuple(i for group in self.physical_groups.values() for i in group)
        basis = ReducedPhysicalBasis(tuple(map(str, self.physical_groups)),
            tuple(self.physical_groups), tuple(map(len, self.physical_groups.values())))
        self.reduced_symmetry = ReducedSymmetry(basis, self.to_sector(symmetry.identity),
                                               self.to_sector(symmetry.sector), name=symmetry.name)

    def to_sector(self, charge):
        values = self.symmetry._normalize_vector(charge, self.symmetry._moduli)
        return Sector(self.labels, values+(SU2Irrep(0),))

    def from_sector(self, sector):
        if not isinstance(sector, Sector) or sector.labels != self.labels or sector.components[-1] != SU2Irrep(0):
            raise ValueError('sector does not belong to this Abelian coordinate map')
        return self.symmetry._external(sector.components[:-1])

    @staticmethod
    def _groups(sectors):
        groups = {}
        for i, sector in enumerate(sectors): groups.setdefault(sector, []).append(i)
        return groups

    def to_reduced(self, state):
        if not isinstance(state, LatticeLETTA) or state.symmetry != self.symmetry:
            raise ValueError('state and Abelian coordinate map do not match')
        if state.symmetry_violation() > 1e-13:
            raise ValueError('cannot convert a state that violates its charge constraints')
        bonds = tuple(tuple(map(self.to_sector, bond)) for bond in state.bond_charges)
        cores = []
        for site, tensor in enumerate(state.tensors):
            left = (self.reduced_symmetry.identity,) if site == 0 else bonds[site-1]
            right = (self.reduced_symmetry.sector,) if site == state.nsites-1 else bonds[site]
            dependencies = [self.physical_order]*(len(state.site_neighborhood(site))-1)
            blocks = {}
            for ql, li in self._groups(left).items():
                for qp, pi in self.physical_groups.items():
                    for qr, ri in self._groups(right).items():
                        if qr in ql.fuse(qp):
                            blocks[ql, qp, qr] = tensor[np.ix_(li, pi, *dependencies, ri)].copy()
            cores.append(blocks)
        return ReducedLatticeLETTA(state.lattice_shape, self.reduced_symmetry, cores,
            bond_sectors=bonds, coordinates=state.coordinates,
            neighborhoods=tuple(state.site_neighborhood(i) for i in range(state.nsites)), normalize=False)

    def from_reduced(self, state):
        if not isinstance(state, ReducedLatticeLETTA) or state.symmetry != self.reduced_symmetry:
            raise ValueError('reduced state and Abelian coordinate map do not match')
        bonds = tuple(tuple(map(self.from_sector, bond)) for bond in state.bond_sectors)
        cores = []
        for site, blocks in enumerate(state.tensors):
            left = (state.symmetry.identity,) if site == 0 else state.bond_sectors[site-1]
            right = (state.symmetry.sector,) if site == state.nsites-1 else state.bond_sectors[site]
            nphys = len(state.site_neighborhood(site))
            dtype = np.result_type(*[a.dtype for a in blocks.values()])
            core = np.zeros((len(left),)+(len(self.physical_order),)*nphys+(len(right),), dtype=dtype)
            lg, rg = self._groups(left), self._groups(right)
            dependencies = [self.physical_order]*(nphys-1)
            for (ql, qp, qr), block in blocks.items():
                core[np.ix_(lg[ql], self.physical_groups[qp], *dependencies, rg[qr])] = block
            cores.append(core)
        return LatticeLETTA(state.lattice_shape, len(self.physical_order), cores,
            coordinates=state.coordinates, neighborhoods=tuple(state.site_neighborhood(i) for i in range(state.nsites)),
            symmetry=self.symmetry, bond_charges=bonds, normalize=False)

    def hamiltonian(self, hamiltonian):
        if not isinstance(hamiltonian, LatticeMPO) or hamiltonian.physical_dim != len(self.physical_order):
            raise TypeError('Abelian native sweeps require a matching LatticeMPO')
        factors = tuple(w[:, :, self.physical_order, :][:, :, :, self.physical_order]
                        for w in hamiltonian.factors)
        result = ReducedMPOHamiltonian(None, factors, name='Abelian MPO')
        # Validate conservation before any optimization, not as a numerical
        # recovery condition at a later site.
        result.native_mpo(self.reduced_symmetry.physical_basis)
        return result


def abelian_dmrg(hamiltonian, *, state, options, bond_dim=None):
    """Run the common one-site/CBE or two-site solver in U(1) blocks.

    The options class selects the update method. The returned state uses the
    original Abelian basis and site/tie ordering, including newly allocated
    charge sectors. D counts ordinary states because every irrep has dimension 1.
    """
    from .solver import LETTADMROptions, letta_dmrg
    from .._letta_two_site_opt.solver import LETTATwoSiteOptions
    from .._letta_two_site_opt.solver import letta_two_site_dmrg
    mapping = AbelianReducedMap(state.symmetry)
    reduced, mpo = mapping.to_reduced(state), mapping.hamiltonian(hamiltonian)
    if isinstance(options, LETTADMROptions):
        if bond_dim is not None:
            raise ValueError('one-site/CBE uses the initial state bond cap, not bond_dim')
        if options.bond_dimension_schedule is not None:
            raise ValueError('shared Abelian bond schedules are not implemented yet')
        result = letta_dmrg(mpo, state=reduced, options=options)
    elif isinstance(options, LETTATwoSiteOptions):
        cap = max(map(len, state.bond_charges), default=1) if bond_dim is None else bond_dim
        result = letta_two_site_dmrg(mpo, state=reduced, bond_dim=cap, options=options)
    else:
        raise TypeError('options must select the one-site/CBE or two-site solver')
    return replace(result, state=mapping.from_reduced(result.state))
