"""Whole-sector ring bond growth and factor installation, including closure."""
from collections import Counter
from operator import index

import numpy as np

from .._letta_one_site_opt.reduced_symmetry import _fuse_sectors
from .._letta_one_site_opt.reduced_ring_state import ReducedRingLETTA
from .reduced_solver import _shrink_source_blocks


def ring_edge(state, left_site):
    left = index(left_site)
    if not 0 <= left <= state.nsites:
        raise IndexError('ring edge out of range')
    return left, (left+1) % (state.nsites+1)


def _physical(state, site):
    return (Counter(state.closure.qns[1]) if site == state.nsites else
            dict(zip(state.physical_basis.sectors, state.physical_basis.multiplicities)))


def _dependencies(state, site):
    return () if site == state.nsites else (state.physical_dim,)*(len(state.site_neighborhood(site))-1)


def _with_pair(state, left, right, bond, blocks):
    """Validate a new state before making any changes visible to the caller."""
    tensors = list(state.tensors)
    closure = state.closure.copy()
    bonds = list(state.bond_sectors)
    bonds[right] = tuple(bond)
    for site, data in zip((left, right), blocks):
        if site == state.nsites:
            closure.data = data
        else:
            tensors[site] = data
    closure.qns = [bonds[-1], tuple(closure.qns[1]), bonds[0]]
    return ReducedRingLETTA(state.physical_basis, state.symmetry.sector, tensors,
        bond_sectors=bonds, closure=closure, neighborhoods=state.neighborhoods, normalize=False)


def grow_ring_bond(state, left_site, multiplicities, *, seed=1701):
    """Pad a bond exactly, with complementary right rows seeded for ALS.

    The left columns are zero, so seeded right rows do not change the state.
    This is allocation plumbing; selecting CBE directions is a separate step.
    """
    left, right = ring_edge(state, left_site)
    old = Counter(state.bond_sectors[right])
    counts = {q: index(n) for q, n in multiplicities.items()}
    if any(n < old[q] or n < 0 for q, n in counts.items()) or any(counts.get(q, 0) < n for q, n in old.items()):
        raise ValueError('growth cannot discard existing ring multiplets')
    counts = {q: n for q, n in counts.items() if n}
    outer_left = Counter(state.bond_sectors[left])
    outer_right = Counter(state.bond_sectors[(right+1) % (state.nsites+1)])
    allowed_left = {qm for ql in outer_left for qp in _physical(state, left)
                    for qm in _fuse_sectors(ql, qp)}
    allowed = {qm for qm in allowed_left if any(qr in outer_right
               for qp in _physical(state, right) for qr in _fuse_sectors(qm, qp))}
    if not counts or set(counts)-allowed:
        raise ValueError('ring growth contains unreachable middle sectors')
    bond = tuple(q for q in sorted(counts) for _ in range(counts[q]))
    rng = np.random.default_rng(seed+left)
    blocks = []
    for side, site in enumerate((left, right)):
        dl, dr = (outer_left, counts) if side == 0 else (counts, outer_right)
        original = state.site_blocks(site)
        dtype = np.result_type(*[a.dtype for a in original.values()])
        data = {}
        for ql, nl in dl.items():
            for qp, np_ in _physical(state, site).items():
                for qr in _fuse_sectors(ql, qp):
                    if qr not in dr:
                        continue
                    key = ql, qp, qr
                    shape = (nl, np_)+_dependencies(state, site)+(dr[qr],)
                    a = np.zeros(shape, dtype=dtype)
                    if key in original:
                        previous = original[key]
                        a[tuple(slice(0, n) for n in previous.shape)] = previous
                    if side == 1 and nl > old[ql]:
                        added = a[old[ql]:]
                        values = rng.normal(size=added.shape)
                        if np.issubdtype(dtype, np.complexfloating):
                            values = values+1j*rng.normal(size=added.shape)
                        added[...] = values/np.sqrt(max(1, added.size))
                    data[key] = a
        blocks.append(data)
    return _with_pair(state, left, right, bond, blocks)


def expand_ring_pair_space(state, left_site, bond_dim):
    """Open locally reachable middle sectors for a two-site solve.

    Temporary capacities are per sector, preserving the incumbent even when
    its allocation exceeds the requested final cap. Compression enforces the
    final total number of multiplets.
    """
    left, right = ring_edge(state, left_site)
    cap = index(bond_dim)
    if cap < 1:
        raise ValueError('ring bond cap must be positive')
    lc, rc = Counter(), Counter()
    for ql, dl in Counter(state.bond_sectors[left]).items():
        for qp, dp in _physical(state, left).items():
            for qm in _fuse_sectors(ql, qp):
                lc[qm] += dl*dp
    for qm in lc:
        for qp, dp in _physical(state, right).items():
            for qr in _fuse_sectors(qm, qp):
                rc[qm] += dp*Counter(state.bond_sectors[(right+1) % (state.nsites+1)])[qr]
    counts = Counter(state.bond_sectors[right])
    lm, rm = int(np.prod(_dependencies(state, left))), int(np.prod(_dependencies(state, right)))
    for qm in lc.keys() & rc.keys():
        if rc[qm]:
            counts[qm] = max(counts[qm], min(cap, lc[qm]*lm, rc[qm]*rm))
    return grow_ring_bond(state, left, counts)


def install_ring_factors(state, problem, left, right, retained):
    """Build a state from compressed factors, physically removing unused copies."""
    i, j = ring_edge(state, problem.left_site)
    available = Counter(state.bond_sectors[j])
    ranks = {q: index(n) for q, n in retained.items()}
    if not any(ranks.values()) or any(n < 0 or n > available[q] for q, n in ranks.items()):
        raise ValueError('invalid retained ring allocation')
    bond = tuple(q for q in sorted(ranks) for _ in range(ranks[q]))
    blocks = [_shrink_source_blocks(embedding.unpack_source(vector), ranks, side)
              for embedding, vector, side in ((problem.left_embedding, left, 'left'),
                                              (problem.right_embedding, right, 'right'))]
    return _with_pair(state, i, j, bond, blocks)
