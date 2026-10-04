"""Small-active-space electronic Hamiltonians for the finite LETTA solver.

Spatial basis: |0>, |alpha>, |beta>, |alpha beta> with alpha before beta.
Integrals are real orthonormal-orbital chemists' (pq|rs), in Hartree.
The exact sum-of-products MPO is intentionally a small-system baseline;
its compressed intermediate storage is checked before allocation. No determinant-space
Hamiltonian or particle-number penalty enters the optimization path.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
from operator import index

import numpy as np

from .state import LatticeLETTA
from .symmetry import AbelianSymmetry
from .operators import LatticeMPO


OCCUPATIONS = np.array([(0, 0), (1, 0), (0, 1), (1, 1)])


def orbital_permutation(order, norb):
    if order is None:
        return tuple(range(norb))
    try:
        order = tuple(index(i) for i in order)
    except TypeError as error:
        raise ValueError('orbital order must contain integers') from error
    if len(order) != norb or set(order) != set(range(norb)):
        raise ValueError('orbital order must be a permutation of 0..norb-1')
    return order


def electron_sector(nelec, norb):
    try:
        alpha, beta = (index(i) for i in nelec)
    except (TypeError, ValueError) as error:
        raise ValueError('nelec must be (nalpha, nbeta)') from error
    if not (0 <= alpha <= norb and 0 <= beta <= norb):
        raise ValueError('electron counts must lie between zero and norb')
    return alpha, beta


def spatial_operators():
    alpha, beta = np.zeros((4, 4)), np.zeros((4, 4))
    alpha[0, 1] = alpha[2, 3] = 1.
    beta[0, 2], beta[1, 3] = 1., -1.
    return np.eye(4), np.diag([1., -1., -1., 1.]), (alpha, beta)


def _fermion_product(norb, operators):
    identity, parity, annihilators = spatial_operators()
    local = [identity.copy() for _ in range(norb)]
    for site, spin, creation in operators:
        for earlier in range(site):
            local[earlier] = local[earlier] @ parity
        a = annihilators[spin]
        local[site] = local[site] @ (a.T if creation else a)
    return local


@dataclass(frozen=True)
class ElectronicProblem:
    h1: np.ndarray
    eri: np.ndarray
    nelec: tuple[int, int]
    ecore: float = 0.
    orbital_order: tuple[int, ...] | None = None

    def __post_init__(self):
        h, g = np.asarray(self.h1), np.asarray(self.eri)
        if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] < 1:
            raise ValueError('h1 must be a nonempty square matrix')
        n = len(h)
        if g.shape != (n,) * 4:
            raise ValueError('eri must have shape (norb,)*4')
        if np.iscomplexobj(h) or np.iscomplexobj(g):
            raise ValueError('this initial chemistry adapter requires real integrals')
        if not np.all(np.isfinite(h)) or not np.all(np.isfinite(g)):
            raise ValueError('integrals must be finite')
        if not np.allclose(h, h.T, atol=1e-12, rtol=1e-12):
            raise ValueError('h1 must be symmetric')
        for other in (g.swapaxes(0, 1), g.swapaxes(2, 3), g.transpose(2, 3, 0, 1)):
            if not np.allclose(g, other, atol=1e-12, rtol=1e-12):
                raise ValueError("eri must have real chemists' integral symmetries")
        if not np.isfinite(self.ecore):
            raise ValueError('ecore must be finite')
        order = orbital_permutation(self.orbital_order, n)
        h, g = np.array(h, dtype=float, copy=True), np.array(g, dtype=float, copy=True)
        h.setflags(write=False); g.setflags(write=False)
        object.__setattr__(self, 'h1', h)
        object.__setattr__(self, 'eri', g)
        object.__setattr__(self, 'nelec', electron_sector(self.nelec, n))
        object.__setattr__(self, 'ecore', float(self.ecore))
        object.__setattr__(self, 'orbital_order', order)

    @property
    def norb(self):
        return len(self.h1)

    def symmetry(self, mode='n_sz', *, two_s=None):
        """Select exact N, (N,2Sz), or (N,2S) spatial-orbital sectors.

        For SU(2), the default spin is ``abs(nalpha-nbeta)``. An explicit
        higher spin must contain that M component and fit the orbital space.
        Number-only mode deliberately permits all spin projections.
        """
        na, nb = self.nelec
        if mode != 'su2' and two_s is not None:
            raise ValueError('two_s is only meaningful for su2 symmetry')
        if mode == 'nalpha_nbeta':
            return AbelianSymmetry(tuple(map(tuple, OCCUPATIONS.tolist())), self.nelec,
                                   moduli=(None, None), name='Nalpha x Nbeta')
        if mode == 'n':
            return AbelianSymmetry((0, 1, 1, 2), na+nb, name='U(1) N')
        if mode == 'n_sz':
            return AbelianSymmetry(((0, 0), (1, 1), (1, -1), (2, 0)),
                                   (na+nb, na-nb), moduli=(None, None),
                                   name='U(1) N x U(1) Sz')
        if mode == 'su2':
            from .reduced_symmetry import ReducedPhysicalBasis, ReducedSymmetry
            two_s = abs(na-nb) if two_s is None else index(two_s)
            if (two_s < abs(na-nb) or two_s > min(na+nb, 2*self.norb-na-nb)
                    or (two_s-na-nb) % 2):
                raise ValueError('target spin is incompatible with electrons and spin projection')
            return ReducedSymmetry.su2(ReducedPhysicalBasis.spatial_orbital(),
                target_charge=na+nb, target_two_j=two_s, name='U(1) N x SU(2)')
        raise ValueError('symmetry must be n, n_sz, nalpha_nbeta, or su2')

    def su2_mpo(self, *, cutoff=0.):
        """Chemistry MPO for exact reduced-spin LETTA states.

        Reuses SU(2) DMRG's chemistry builder and reduced AutoMPO. Its exact
        local magnetic-component factors supply an operator reference. Native
        sweeps compile these once into irreducible operator multiplets, then
        contract multiplicities only. ``factors=None`` avoids treating the
        builder's unverified reduced view as that compiled representation.
        The scalar offset is included exactly once, including one-orbital cases.
        """
        from .reduced_operators import ReducedMPOHamiltonian, physical_leg_from_reduced_basis
        from .reduced_symmetry import ReducedPhysicalBasis
        from pyqed.mps.nonabelian.builder import AutoMPO, identity_operator
        from pyqed.mps.nonabelian.mpo import MPO, SiteOperator, sum_mpo_chains
        from pyqed.mps.nonabelian.models import add_spatial_one_body_terms, SpatialSpinFreeERIBuilder

        if not np.isfinite(cutoff) or cutoff < 0:
            raise ValueError('cutoff must be finite and nonnegative')
        leg = physical_leg_from_reduced_basis(ReducedPhysicalBasis.spatial_orbital(),
                                             fully_reduced=False)
        if self.norb == 1:
            # N and n_up*n_down are scalars; no AutoMPO chain is necessary.
            h = self.h1[0, 0] if abs(self.h1[0, 0]) > cutoff else 0.
            u = self.eri[0, 0, 0, 0] if abs(self.eri[0, 0, 0, 0]/2) > cutoff else 0.
            local = np.diag([self.ecore, self.ecore+h, self.ecore+h, self.ecore+2*h+u])
            factors = [MPO.from_site_operator(SiteOperator.from_dense(local, phys_out_leg=leg, phys_in_leg=leg))]
        else:
            # Match SpatialReducedHamiltonianBuilder's separated assembly;
            # importing its molecular frontend would require gbasis even for
            # already supplied integrals. Shared AutoMPO prefix states must
            # not merge the one-body and ERI-correction operator families.
            auto = AutoMPO([leg]*self.norb)
            add_spatial_one_body_terms(auto, self.h1, cutoff=cutoff)
            eri_factors = SpatialSpinFreeERIBuilder(
                [leg]*self.norb, self.eri, cutoff=cutoff).build()
            factors = sum_mpo_chains(auto.build(), eri_factors, phys_leg=leg)
            if self.ecore or not factors:
                scalar = [MPO.from_site_operator(identity_operator(leg)) for _ in range(self.norb)]
                local = self.ecore*np.eye(4)
                scalar[0] = MPO.from_site_operator(SiteOperator.from_dense(local, phys_out_leg=leg, phys_in_leg=leg))
                factors = sum_mpo_chains(factors, scalar, phys_leg=leg)
        return ReducedMPOHamiltonian(None, tuple(factors), name='SU(2) electronic Hamiltonian')

    def reordered(self, order):
        """Reorder current sites; returned labels refer to the original orbitals."""
        p = orbital_permutation(order, self.norb)
        return ElectronicProblem(self.h1[np.ix_(p, p)], self.eri[np.ix_(p, p, p, p)],
                                 self.nelec, self.ecore,
                                 tuple(self.orbital_order[i] for i in p))

    def mpo(self, *, cutoff=0., max_storage_bytes=256 * 1024**2, backend="svd"):
        """H = ecore + h_pq a†_pσ a_qσ + (pq|rs)/2 a†_pσ a†_rτ a_sτ a_qσ.

        Identical strings are combined. The default ``svd`` backend compresses
        the product sum, removing floating-point null singular directions
        (eps * matrix size * largest value). ``symbolic`` uses pyqed's graph
        AutoMPO builder and avoids the large intermediate SVD frames.
        The storage budget guards product collection and SVD frames; it is not
        a bound on the symbolic builder's total working memory.
        ``cutoff=0`` retains all nonzero coefficients. A positive cutoff is
        an explicit Hamiltonian approximation, not an eigensolver tolerance.
        """
        if backend not in {'svd', 'symbolic'}:
            raise ValueError('MPO backend must be svd or symbolic')
        if not np.isfinite(cutoff) or cutoff < 0:
            raise ValueError('cutoff must be finite and nonnegative')
        max_storage_bytes = index(max_storage_bytes)
        if max_storage_bytes < 1:
            raise ValueError('storage budget must be positive')
        identity, _, _ = spatial_operators()
        terms = {}
        def add(coefficient, operators):
            if abs(coefficient) <= cutoff:
                return
            local = _fermion_product(self.norb, operators)
            if any(not np.any(a) for a in local):
                return
            # Canonicalize signs so cancellations combine the same product.
            for a in local:
                first = a.flat[np.flatnonzero(a)[0]]
                coefficient *= first
                a /= first
            key = tuple(a.tobytes() for a in local)
            if key in terms:
                terms[key][0] += coefficient
            else:
                if (len(terms)+1)*self.norb*16*8*4 > max_storage_bytes:
                    raise ValueError('sum-of-products MPO exceeds storage budget')
                terms[key] = [coefficient, local]
        add(self.ecore, [])
        for p, q in np.argwhere(np.abs(self.h1) > cutoff):
            for spin in range(2):
                add(self.h1[p, q], [(p, spin, True), (q, spin, False)])
        for p, q, r, s in np.argwhere(np.abs(self.eri) > 2 * cutoff):
            for spin in range(2):
                for tau in range(2):
                    add(.5 * self.eri[p, q, r, s],
                        [(p, spin, True), (r, tau, True), (s, tau, False), (q, spin, False)])
        products = [(c, local) for c, local in terms.values() if abs(c) > cutoff]
        if not products:
            return LatticeMPO([np.zeros((1,1,4,4))] +
                              [identity.reshape(1,1,4,4)]*(self.norb-1),
                              lattice_shape=(1,self.norb))
        if backend == 'symbolic':
            return _symbolic_product_mpo(products, self.norb)
        # Compress the CP sum directly, from right to left. Never allocate
        # diagonal channel MPOs of shape (nterms,nterms,4,4), or a 4^n matrix.
        coefficients = np.array([c for c, _ in products])
        right = np.ones((len(products), 1))
        factors = []
        from scipy.linalg import svd
        for site in range(self.norb-1, 0, -1):
            width = right.shape[1]
            entries = len(products)*16*width
            if entries*8*5 > max_storage_bytes:
                raise ValueError('sum-of-products MPO exceeds storage budget; use a compact chemistry MPO for this active space')
            local = np.array([a[site].ravel() for _, a in products])
            matrix = (local[:,:,None]*right[:,None,:]).reshape(len(products),16*width)
            u, singular, vh = svd(matrix, full_matrices=False)
            # Remove only numerical null directions of the exact product sum.
            threshold = np.finfo(float).eps*max(matrix.shape)*singular[0]
            rank = max(1, int(np.count_nonzero(singular > threshold)))
            factors.append(vh[:rank].reshape(rank,4,4,width).transpose(0,3,1,2))
            right = u[:,:rank]*singular[:rank]
        if self.norb == 1 and 16*8 > max_storage_bytes:
            raise ValueError('sum-of-products MPO exceeds storage budget')
        local = np.array([a[0].ravel() for _, a in products])
        first = np.einsum('t,tp,tr->rp', coefficients, local, right).reshape(1,-1,4,4)
        factors.append(first)
        factors.reverse()
        # The right basis spans all suffix strings, including combinations
        # absent from this Hamiltonian. Sweep left to remove that redundancy.
        for site in range(self.norb-1):
            a = factors[site]
            matrix = a.transpose(0,2,3,1).reshape(a.shape[0]*16,a.shape[1])
            u, singular, vh = svd(matrix, full_matrices=False)
            threshold = np.finfo(float).eps*max(matrix.shape)*singular[0]
            rank = max(1, int(np.count_nonzero(singular > threshold)))
            factors[site] = u[:,:rank].reshape(a.shape[0],4,4,rank).transpose(0,3,1,2)
            transfer = singular[:rank,None]*vh[:rank]
            factors[site+1] = np.tensordot(transfer,factors[site+1],axes=(1,0))
        return LatticeMPO(factors, lattice_shape=(1,self.norb))



def sector_bond_charges(norb, nelec, max_bond_dim):
    """Allocate every reachable (Nalpha,Nbeta) sector before multiplicities.

    The cap is the total dimension of each virtual bond. Reject too-small caps
    instead of silently discarding whole charge sectors. Multiplicities are
    bounded by the smaller left/right determinant counts for ordinary MPS.
    """
    norb, cap = index(norb), index(max_bond_dim)
    if norb < 1 or cap < 1:
        raise ValueError('norb and max_bond_dim must be positive')
    na, nb = electron_sector(nelec, norb)
    bonds = []
    for cut in range(1, norb):
        remaining = norb - cut
        sectors = [(a, b) for a in range(max(0, na-remaining), min(na, cut)+1)
                   for b in range(max(0, nb-remaining), min(nb, cut)+1)]
        if cap < len(sectors):
            raise ValueError(f'bond {cut} needs at least {len(sectors)} states to retain every charge sector')
        capacity = [min(comb(cut, a)*comb(cut, b),
                        comb(remaining, na-a)*comb(remaining, nb-b)) for a, b in sectors]
        counts = [1] * len(sectors)
        for _ in range(min(cap, sum(capacity)) - len(sectors)):
            candidates = [i for i in range(len(sectors)) if counts[i] < capacity[i]]
            chosen = max(candidates, key=lambda i: (capacity[i]/counts[i], -i))
            counts[chosen] += 1
        bonds.append(tuple(q for q, copies in zip(sectors, counts) for _ in range(copies)))
    return tuple(bonds)


def initial_state(problem, *, max_bond_dim=16, seed=731, symmetry='nalpha_nbeta'):
    """Random charge-complete MPS; embed the same state into chosen ties later."""
    symmetry = problem.symmetry(symmetry)
    if not isinstance(symmetry, AbelianSymmetry):
        raise ValueError('use ReducedLatticeLETTA.random or from_mps for SU(2) states')
    bonds = abelian_sector_bond_charges(problem.norb, symmetry, max_bond_dim)
    dims = (1,) + tuple(map(len, bonds)) + (1,)
    rng = np.random.default_rng(seed)
    tensors = [rng.normal(size=(dims[i], 4, dims[i+1])) for i in range(problem.norb)]
    return LatticeLETTA((1, problem.norb), 4, tensors,
                       neighborhoods=tuple((i,) for i in range(problem.norb)),
                       symmetry=symmetry, bond_charges=bonds)


def abelian_sector_bond_charges(nsites, symmetry, max_bond_dim):
    """Charge-complete allocation, capped by left/right MPS sector ranks.

    Repeated physical charges (up/down in N-only mode) count separately.
    This is an initialization, not adaptive sector optimization.
    """
    from collections import Counter
    nsites, cap = index(nsites), index(max_bond_dim)
    if nsites < 1 or cap < 1:
        raise ValueError('nsites and max_bond_dim must be positive')
    counts = [Counter({symmetry.identity: 1})]
    for _ in range(nsites):
        next_counts = Counter()
        for q, count in counts[-1].items():
            for p in symmetry.physical_charges:
                next_counts[symmetry.fuse(q, p)] += count
        counts.append(next_counts)
    if not counts[-1][symmetry.sector]:
        raise ValueError('unreachable electron sector')
    bonds = []
    for cut in range(1, nsites):
        capacities = {q: min(n, counts[nsites-cut][symmetry.difference(symmetry.sector, q)])
                      for q, n in counts[cut].items()}
        sectors = sorted(q for q, n in capacities.items() if n)
        if cap < len(sectors):
            raise ValueError(f'bond {cut} needs at least {len(sectors)} states to retain every charge sector')
        copies = {q: 1 for q in sectors}
        for _ in range(min(cap, sum(capacities.values()))-len(sectors)):
            q = max((q for q in sectors if copies[q] < capacities[q]),
                    key=lambda q: capacities[q]/copies[q])
            copies[q] += 1
        bonds.append(tuple(q for q in sectors for _ in range(copies[q])))
    return tuple(bonds)


def embed_ties(mps, neighborhoods):
    """Embed an untied MPS exactly, without introducing tie-dependent noise."""
    if mps.physical_dim != 4:
        raise ValueError('chemistry ties require four-state spatial orbitals')
    if any(mps.site_neighborhood(i) != (i,) for i in range(mps.nsites)):
        raise ValueError('the initial state must be an untied MPS')
    from .state import _validate_neighborhoods
    neighborhoods = _validate_neighborhoods(neighborhoods, mps.nsites)
    tensors = []
    for i, sites in enumerate(neighborhoods):
        a = mps.tensors[i]
        shape = (a.shape[0], 4) + (1,) * (len(sites)-1) + (a.shape[-1],)
        full = (a.shape[0],) + (4,) * len(sites) + (a.shape[-1],)
        tensors.append(np.broadcast_to(a.reshape(shape), full).copy())
    return LatticeLETTA(mps.lattice_shape, 4, tensors, neighborhoods=neighborhoods,
                       symmetry=mps.symmetry, bond_charges=mps.bond_charges)


def _symbolic_product_mpo(products, norb):
    """Use pyqed's graph-based AutoMPO builder without dense CP SVD frames."""
    from pyqed.mps.autompo.basis import BasisSet
    from pyqed.mps.autompo.Operator import Op
    from pyqed.mps.autompo.model import Model
    from pyqed.mps.autompo.light_automatic_mpo import Mpo

    aliases = {'I': np.eye(4)}
    names = {aliases['I'].tobytes(): 'I'}
    terms = []
    for coefficient, matrices in products:
        symbols, sites = [], []
        for site, matrix in enumerate(matrices):
            key = matrix.tobytes()
            if key not in names:
                name = f'local{len(names)}'
                names[key] = name
                aliases[name] = matrix
            name = names[key]
            if name != 'I':
                symbols.append(name); sites.append(site)
        terms.append(Op(' '.join(symbols) if symbols else 'I', sites if sites else [0],
                        factor=coefficient))

    class SpatialOperatorBasis(BasisSet):
        def __init__(self, site):
            super().__init__(site, 4, [0]*4)

        def op_mat(self, op):
            result = np.eye(4)
            for symbol in op.split_symbol:
                result = result @ aliases[symbol]
            return result * op.factor

        def copy(self, new_dof):
            return SpatialOperatorBasis(new_dof)

    model = Model([SpatialOperatorBasis(i) for i in range(norb)], terms)
    mpo = Mpo(model, algo='Hopcroft-Karp')
    return LatticeMPO([w.transpose(0,3,1,2) for w in mpo.matrices],
                      lattice_shape=(1,norb))
