"""Reduced conditional LETTA on a virtual ring with a covariant target core."""
from collections import Counter
from operator import index

import numpy as np

from pyqed.mps.su2 import SU2Irrep
from pyqed.mps.symmetry import Sector
from .state import _validate_neighborhoods
from .reduced_state import ReducedLatticeLETTA
from .reduced_symmetry import ReducedSymmetry, _fuse_sectors
from .reduced_ring_target import ReducedRingTarget, signed_sector, signed_physical_basis, dual_sector
from .reduced_ring_contraction import CyclicReducedNorm


class ReducedRingLETTA:
    """Physical conditional cores plus one explicit dual-target closure.

    ``bond_sectors`` has L+1 entries: before physical core zero, then after
    every physical core. The closure connects the last entry back to the first.
    Dependencies refer only to physical sites. Repeated labels are reduced
    multiplet/copy labels, never magnetic spin components.
    """
    virtual_topology = 'closed'

    def __init__(self, physical_basis, target, tensors, *, bond_sectors,
                 closure, neighborhoods=None, normalize=True):
        tensors = tuple(tensors)
        self.nsites = len(tensors)
        if self.nsites < 1:
            raise ValueError('a ring needs at least one physical core')
        self.physical_basis = physical_basis
        target = signed_sector(target)
        identity = Sector(target.labels, tuple(SU2Irrep(0) if label == 'su2' else 0
                                               for label in target.labels))
        self.symmetry = ReducedSymmetry(physical_basis, identity, target)
        self._neighborhoods = _validate_neighborhoods(
            tuple((i,) for i in range(self.nsites)) if neighborhoods is None else neighborhoods,
            self.nsites)
        self.bond_sectors = tuple(tuple(bond) for bond in bond_sectors)
        if len(self.bond_sectors) != self.nsites+1 or any(not b for b in self.bond_sectors):
            raise ValueError('ring requires L+1 nonempty virtual allocations')
        self.closure = closure.copy()
        if (Counter(self.closure.qns[0]) != Counter(self.bond_sectors[-1]) or
                Counter(self.closure.qns[2]) != Counter(self.bond_sectors[0])):
            raise ValueError('target closure does not match ring endpoint allocations')
        self.tensors = ReducedLatticeLETTA._validate_tensors(self, tensors)
        self.to_target_ring()  # Validate the closure, fusion rules and actual loop.
        if normalize:
            self.normalize()

    @classmethod
    def random(cls, nsites, physical_basis, target, *, multiplets_per_sector=1,
               anchor_sector=None, neighborhoods=None, seed=None, real=False):
        """Seed all fusion-compatible sectors with a chosen copy count.

        The anchor is a virtual irrep at the distinguished closure cut, not
        a physical boundary vacuum. Its multiplicity also equals the requested
        copy count. This option counts copies PER sector, not a total D cap.
        Explicit allocations can instead be supplied to the constructor.
        """
        from pyqed.mps.nonabelian.tensor import NonabelianTensor
        nsites, copies = index(nsites), index(multiplets_per_sector)
        if nsites < 1 or copies < 1:
            raise ValueError('site and multiplet-copy counts must be positive')
        basis, target = signed_physical_basis(physical_basis), signed_sector(target)
        if anchor_sector is None:
            anchor = Sector(target.labels, tuple(SU2Irrep(0) if label == 'su2' else 0
                                                  for label in target.labels))
        else:
            anchor = signed_sector(anchor_sector)
        neighborhoods = _validate_neighborhoods(
            tuple((i,) for i in range(nsites)) if neighborhoods is None else neighborhoods, nsites)
        forward = [{anchor}]
        for i in range(nsites):
            forward.append({qr for ql in forward[-1] for qp in basis.sectors
                            for qr in _fuse_sectors(ql, qp)})
        dual = dual_sector(target)
        forward[-1] = {q for q in forward[-1] if anchor in _fuse_sectors(q, dual)}
        for i in range(nsites-1, -1, -1):
            forward[i] = {q for q in forward[i] if any(qr in forward[i+1]
                for qp in basis.sectors for qr in _fuse_sectors(q, qp))}
        if not forward[0]:
            raise ValueError('target is unreachable for this ring basis and anchor')
        bonds = tuple(tuple(q for q in sorted(layer) for _ in range(copies)) for layer in forward)
        rng = np.random.default_rng(seed)
        tensors = []
        for i in range(nsites+1):
            physical = ((dual, 1),) if i == nsites else zip(basis.sectors, basis.multiplicities)
            left, right = bonds[i], bonds[(i+1) % (nsites+1)]
            dependencies = () if i == nsites else (basis.reduced_dim,)*(len(neighborhoods[i])-1)
            data = {}
            for qp, dp in physical:
                for ql, dl in Counter(left).items():
                    for qr, dr in Counter(right).items():
                        if qr in _fuse_sectors(ql, qp):
                            shape = (dl, dp)+dependencies+(dr,)
                            values = rng.normal(size=shape)
                            if not real:
                                values = values+1j*rng.normal(size=shape)
                            data[ql, qp, qr] = values/np.sqrt(np.prod(shape))
            tensors.append(data)
        closure = NonabelianTensor(data=tensors.pop(), qns=[bonds[-1], (dual,), bonds[0]],
            dirs=[-1, 1, 1], metadata={'physical_basis': 'fully_reduced_su2'})
        return cls(basis, target, tensors, bond_sectors=bonds, closure=closure,
                   neighborhoods=neighborhoods, normalize=True)

    @classmethod
    def from_target_ring(cls, ring, physical_basis, *, neighborhoods=None, normalize=False):
        n = len(ring.physical_sites)
        neighborhoods = _validate_neighborhoods(
            tuple((i,) for i in range(n)) if neighborhoods is None else neighborhoods, n)
        physical = Counter(dict(zip(physical_basis.sectors, physical_basis.multiplicities)))
        tensors = []
        for i, site in enumerate(ring.physical_sites):
            if Counter(site.qns[1]) != physical:
                raise ValueError('ring core physical basis mismatch')
            blocks = {}
            for key, a in site.data.items():
                shape = a.shape[:2] + (physical_basis.reduced_dim,)*(len(neighborhoods[i])-1) + a.shape[2:]
                blocks[key] = np.broadcast_to(a.reshape(a.shape[:2]+(1,)*(len(neighborhoods[i])-1)+a.shape[2:]), shape).copy()
            tensors.append(blocks)
        bonds = (tuple(ring.physical_sites[0].qns[0]),) + tuple(tuple(s.qns[2]) for s in ring.physical_sites)
        return cls(physical_basis, ring.target, tensors, bond_sectors=bonds,
                   closure=ring.closure, neighborhoods=neighborhoods, normalize=normalize)

    @property
    def physical_dim(self):
        return self.physical_basis.reduced_dim

    @property
    def neighborhoods(self):
        return self._neighborhoods

    @property
    def bond_dimensions(self):
        return tuple(map(len, self.bond_sectors))

    @property
    def parameter_count(self):
        return sum(a.size for core in self.tensors for a in core.values()) + sum(a.size for a in self.closure.data.values())

    def site_neighborhood(self, site):
        return self._neighborhoods[index(site)]

    def left_virtual_sectors(self, site):
        return self.bond_sectors[site]

    def right_virtual_sectors(self, site):
        return self.bond_sectors[site+1]

    def copy(self):
        return type(self)(self.physical_basis, self.symmetry.sector, self.tensors,
                         bond_sectors=self.bond_sectors, closure=self.closure,
                         neighborhoods=self.neighborhoods, normalize=False)

    def restore(self, other):
        saved = other.copy()
        self.tensors, self.closure, self.bond_sectors = saved.tensors, saved.closure, saved.bond_sectors

    def site_blocks(self, site):
        if not 0 <= site <= self.nsites:
            raise IndexError('ring core index out of range')
        return self.closure.data if site == self.nsites else self.tensors[site]

    def set_site_blocks(self, site, blocks):
        if site == self.nsites:
            self.closure.data = blocks
        elif 0 <= site < self.nsites:
            self.tensors[site] = blocks
        else:
            raise IndexError('ring core index out of range')

    def to_target_ring(self):
        from .reduced_frontier import ReducedFrontier
        sites = ReducedFrontier.from_state(self).to_mps(self)
        return ReducedRingTarget(tuple(sites), self.closure, self.symmetry.sector)

    def norm(self):
        value = CyclicReducedNorm(self.to_target_ring().sites).overlap()
        if (not np.isfinite(value) or value.real <= np.finfo(float).tiny or
                abs(value.imag) > 1e-10*value.real):
            raise FloatingPointError('invalid reduced ring norm')
        return float(np.sqrt(value.real))

    def normalize(self, *, center=0, balance=True):
        snapshot = [{key: a.copy() for key, a in self.site_blocks(i).items()}
                    for i in range(self.nsites+1)]
        try:
            if balance:
                for i in range(self.nsites+1):
                    core = self.site_blocks(i)
                    scale = max(float(np.max(np.abs(a), initial=0.)) for a in core.values())
                    if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
                        raise FloatingPointError('null or nonfinite ring core')
                    self.set_site_blocks(i, {key: a/scale for key, a in core.items()})
            value = self.norm()
            self.set_site_blocks(center, {key: a/value for key, a in self.site_blocks(center).items()})
        except Exception:
            for i, core in enumerate(snapshot):
                self.set_site_blocks(i, core)
            raise
        return self
