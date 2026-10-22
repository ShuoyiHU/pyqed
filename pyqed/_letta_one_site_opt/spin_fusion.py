"""Exact pair blocking that exposes SU(2) spin-fusion labels to LETTA ties."""
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .reduced_mpo_compile import cg_tensor
from .reduced_operators import ReducedMPOHamiltonian
from .reduced_state import ReducedLatticeLETTA
from .reduced_symmetry import ReducedBasisState, ReducedPhysicalBasis, ReducedSymmetry, _fuse_sectors, _sector_irrep


@dataclass(frozen=True)
class CoupledPhysicalPair:
    """Local coupled basis, its fusion paths, and product-to-coupled columns.

    ``transform[:, k]`` is coupled state k in the ordered product basis.
    Each channel is (left reduced state, right reduced state, coupled state).
    Only a two-site physical transformation is stored, never a many-body one.
    """
    basis: ReducedPhysicalBasis
    channels: tuple
    transform: np.ndarray


def coupled_pair_basis(physical_basis):
    """Fuse two identical local bases without discarding any multiplets.

    This uses the standard SU(2) structural/degeneracy decomposition of
    Singh and Vidal, Phys. Rev. B 86, 195114 (2012),
    https://doi.org/10.1103/PhysRevB.86.195114. Its use as LETTA conditioning
    is a rung-blocked adaptation, not an arbitrary intersite spin-tie method.
    For Hubbard orbitals the result has 10 multiplets / 16 magnetic states,
    including distinct singly-occupied-pair singlet and triplet channels.
    """
    if not isinstance(physical_basis, ReducedPhysicalBasis):
        raise TypeError('physical_basis must be ReducedPhysicalBasis')
    groups = defaultdict(list)
    for left in physical_basis.reduced_states:
        for right in physical_basis.reduced_states:
            for sector in _fuse_sectors(left.sector, right.sector):
                groups[sector].append((left, right))
    sectors = tuple(sorted(groups))
    basis = ReducedPhysicalBasis(tuple(str(q) for q in sectors), sectors,
                                 tuple(len(groups[q]) for q in sectors))
    offsets, offset = {}, 0
    for state in physical_basis.reduced_states:
        offsets[state] = offset
        offset += state.irrep.dim
    transform = np.zeros((offset**2, basis.dense_dim))
    channels, column = [], 0
    for q, label in zip(sectors, basis.labels):
        for copy, (left, right) in enumerate(groups[q]):
            output = ReducedBasisState(label, q, copy)
            channels.append((left, right, output))
            cg = cg_tensor(left.irrep, right.irrep, output.irrep)
            for a in range(left.irrep.dim):
                for b in range(right.irrep.dim):
                    row = (offsets[left]+a)*offset + offsets[right]+b
                    transform[row, column:column+output.irrep.dim] = cg[a,b]
            column += output.irrep.dim
    return CoupledPhysicalPair(basis, tuple(channels), transform)


@lru_cache(maxsize=None)
def _recoupling(left, first, middle, second, right, pair):
    # Overlap of ((left first) middle, second) right with
    # (left, (first second) pair) right, averaged over the final multiplet.
    return float(np.einsum('abm,mcr,bcp,apr->',
        cg_tensor(left, first, middle), cg_tensor(middle, second, right),
        cg_tensor(first, second, pair), cg_tensor(left, pair, right),
        optimize=True) / right.dim)


def fuse_su2_pairs(state, hamiltonian, *, neighborhoods=None):
    """Exactly block adjacent untied SU(2) MPS sites, then add optional ties.

    Pairing is (0,1), (2,3), ...; use rung-major orbital ordering for ladders.
    The returned (state, Hamiltonian) uses the existing reduced LETTA solver.
    Neighbor conditioning now sees pair spin and fusion multiplicity as well
    as charge. Input must be untied: existing orbital ties are not discarded.
    Inter-pair backbone multiplets and the initial physical state are retained
    exactly; intra-pair bond restrictions are removed by subsequent updates.
    D therefore refers to inter-pair bonds, not all original orbital cuts.

    Standard SU(2) recoupling follows Singh and Vidal, PRB 86, 195114 (2012),
    https://doi.org/10.1103/PhysRevB.86.195114. This is an exact local basis
    change followed by a restricted, rung-blocked LETTA adaptation. It is not
    a general overlapping spin-fusion tie implementation. Hamiltonian cores
    are transformed locally and then compiled to reduced operators; there is
    no determinant-space projection in the optimization path. No guarantee
    of reaching the global variational minimum is implied.
    """
    if not isinstance(state, ReducedLatticeLETTA):
        raise TypeError('state must be ReducedLatticeLETTA')
    if state.nsites % 2 or state.nsites < 4:
        raise ValueError('pair blocking requires an even number of at least four sites')
    if any(state.site_neighborhood(i) != (i,) for i in range(state.nsites)):
        raise ValueError('pair blocking requires an untied MPS input')
    if not isinstance(hamiltonian, ReducedMPOHamiltonian) or len(hamiltonian) != state.nsites:
        raise ValueError('Hamiltonian must be a matching ReducedMPOHamiltonian')
    pair = coupled_pair_basis(state.physical_basis)
    symmetry = ReducedSymmetry(pair.basis, state.symmetry.identity, state.symmetry.sector,
                               state.symmetry.name)
    count = state.nsites//2
    if neighborhoods is None:
        neighborhoods = tuple((i,) for i in range(count))
    # Validate before allocating conditional axes.
    from .state import _validate_neighborhoods
    neighborhoods = _validate_neighborhoods(neighborhoods, count)
    physical_mult = dict(zip(pair.basis.sectors, pair.basis.multiplicities))
    tensors, cores = [], []
    for i in range(count):
        first, second = state.tensors[2*i:2*i+2]
        blocks = {}
        dtype = np.result_type(*[v.dtype for t in (first, second) for v in t.values()], float)
        for (ql, qa, qm), a in first.items():
            for (middle, qb, qr), b in second.items():
                if middle != qm:
                    continue
                for left, right, output in pair.channels:
                    if left.sector != qa or right.sector != qb or qr not in _fuse_sectors(ql, output.sector):
                        continue
                    coefficient = _recoupling(*map(_sector_irrep, (ql,qa,qm,qb,qr,output.sector)))
                    if coefficient == 0:
                        continue
                    key = (ql, output.sector, qr)
                    if key not in blocks:
                        blocks[key] = np.zeros((a.shape[0],physical_mult[output.sector],b.shape[-1]),
                                               dtype=dtype)
                    blocks[key][:,output.copy,:] += coefficient * (a[:,left.copy,:] @ b[:,right.copy,:])
        nties = len(neighborhoods[i])-1
        tensors.append({key: np.broadcast_to(value.reshape(value.shape[:2]+(1,)*nties+value.shape[2:]),
            value.shape[:2]+(pair.basis.reduced_dim,)*nties+value.shape[2:]).copy()
            for key,value in blocks.items()})
        w, v = hamiltonian.canonical_factors[2*i:2*i+2]
        w = np.asarray(w.as_dense() if hasattr(w,'as_dense') else w)
        v = np.asarray(v.as_dense() if hasattr(v,'as_dense') else v)
        d = state.physical_basis.dense_dim
        if w.shape[2:] != (d,d) or v.shape[2:] != (d,d):
            raise ValueError('Hamiltonian physical dimensions do not match state')
        core = np.einsum('abij,bckl->acikjl',w,v,optimize=True).reshape(w.shape[0],v.shape[1],d*d,d*d)
        cores.append(np.einsum('ox,aboi,iy->abxy',pair.transform.conj(),core,pair.transform,optimize=True))
    blocked = ReducedLatticeLETTA((1,count),symmetry,tensors,
        bond_sectors=state.bond_sectors[1::2],neighborhoods=neighborhoods,normalize=False)
    ham = ReducedMPOHamiltonian(None,tuple(cores),name=hamiltonian.name+' (SU2 pair basis)')
    return blocked, ham
