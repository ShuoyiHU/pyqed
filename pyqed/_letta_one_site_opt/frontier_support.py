"""Exact Boolean reachability for tie-conditioned, open-boundary LETTA.

Nodes carry a backbone symmetry sector and the physical labels pending across
that cut. Forward propagation and backward pruning retain precisely the paths
that connect the identity to the target. This is structural sparsity, with no
coefficient tests, numerical tolerances, or rank approximation. Sector
multiplicity copies remain the caller's responsibility.
"""

from collections import OrderedDict
from hashlib import sha256
from itertools import product


def _signature(state, cuts):
    from .reduced_state import ReducedLatticeLETTA
    if not isinstance(state, ReducedLatticeLETTA):
        raise TypeError('frontier support requires open-boundary ReducedLatticeLETTA')
    cuts = tuple(tuple(int(i) for i in cut) for cut in cuts)
    if len(cuts) != state.nsites - 1:
        raise ValueError('frontier support needs one pending-label list per internal cut')
    if any(len(set(cut)) != len(cut) or any(i < 0 or i >= state.nsites for i in cut)
           for cut in cuts):
        raise ValueError('invalid frontier pending-label list')
    neighborhoods = tuple(tuple(state.site_neighborhood(i)) for i in range(state.nsites))
    physical = tuple((q.sector, q.copy) for q in state.physical_basis.reduced_states)
    shapes = tuple(tuple((key, tuple(block.shape)) for key, block in sorted(tensor.items()))
                   for tensor in state.tensors)
    return (cuts, neighborhoods, physical, state.symmetry.identity, state.symmetry.sector,
            tuple(tuple(bond) for bond in state.bond_sectors), shapes)


class FrontierSupport:
    """Supported cut assignments and local routes, independent of tensor values.

    ``bases[k][sector]`` contains the sorted assignment tuples before site k;
    the two endpoint assignments are empty. Each edge is
    ``(left_sector, physical_sector, right_sector, physical_copy,
    left_assignment, right_assignment, local_labels)``. ``local_labels`` follows
    the site's neighborhood order, including its own physical label first.
    """

    def __init__(self, state, cuts):
        self.signature = _signature(state, cuts)
        self.fingerprint = sha256(repr(self.signature).encode()).hexdigest()
        cuts, neighborhoods, physical, identity, target, _, shapes = self.signature
        pending = ((),) + cuts + ((),)
        state_index = {label: i for i, label in enumerate(physical)}
        radix = len(physical)
        reachable = [{(identity, ())}]
        routes = []
        for site, blocks in enumerate(shapes):
            left_variables, right_variables = pending[site:site + 2]
            if any(v not in set(left_variables) | set(right_variables) | {site}
                   for v in neighborhoods[site]):
                raise ValueError('frontier cuts do not expose every local dependency')
            new_variables = tuple(v for v in right_variables
                                  if v not in left_variables and v != site)
            by_left = {}
            for key, shape in blocks:
                if any(size == 0 for size in shape):
                    continue
                for copy in range(shape[1]):
                    by_left.setdefault(key[0], []).append((key, copy, state_index[key[1], copy]))
            edges, following = [], set()
            for qleft, left in sorted(reachable[-1]):
                values = dict(zip(left_variables, left))
                for key, copy, own in by_left.get(qleft, ()):
                    if site in values and values[site] != own:
                        continue
                    assigned = dict(values)
                    assigned[site] = own
                    for choice in product(range(radix), repeat=len(new_variables)):
                        assigned.update(zip(new_variables, choice))
                        right = tuple(assigned[v] for v in right_variables)
                        local = tuple(assigned[v] for v in neighborhoods[site])
                        edges.append((*key, copy, left, right, local))
                        following.add((key[2], right))
            routes.append(edges)
            reachable.append(following)

        reachable[-1].intersection_update({(target, ())})
        for site in range(state.nsites - 1, -1, -1):
            routes[site] = tuple(edge for edge in routes[site]
                                 if (edge[2], edge[5]) in reachable[site + 1])
            reachable[site] = {(edge[0], edge[4]) for edge in routes[site]}
        bases = []
        for nodes in reachable:
            basis = {}
            for sector, assignment in sorted(nodes):
                basis.setdefault(sector, []).append(assignment)
            bases.append({sector: tuple(assignments) for sector, assignments in basis.items()})
        self.bases = tuple(bases)
        self.edges = tuple(routes)
        # Conservative accounting bounds retained Python routing metadata, not
        # numerical arrays. Large individual graphs are used without caching.
        self.cache_bytes = sum(384 + 16 * (len(e[4]) + len(e[5]) + len(e[6]))
                               for edges in self.edges for e in edges)
        self.cache_bytes += sum(192 + 16 * len(assignment) for basis in self.bases
                                for assignments in basis.values() for assignment in assignments)
        self.cache_bytes += 256 * sum(len(blocks) for blocks in shapes)


_SUPPORT_CACHE = OrderedDict()
_SUPPORT_CACHE_BYTES = 0
_SUPPORT_CACHE_LIMIT = 64 * 1024**2


def get_frontier_support(state, cuts, previous=None):
    """Reuse the caller's current graph or the bounded shared graph cache.

    Keeping one ``previous`` graph lets a frontier reuse large layouts that
    exceed the shared cache budget without recompiling them for every site.
    """
    global _SUPPORT_CACHE_BYTES
    signature = _signature(state, cuts)
    if previous is not None and previous.signature == signature:
        return previous
    support = _SUPPORT_CACHE.get(signature)
    if support is not None:
        _SUPPORT_CACHE.move_to_end(signature)
        return support
    support = FrontierSupport(state, signature[0])
    if support.cache_bytes <= _SUPPORT_CACHE_LIMIT:
        while _SUPPORT_CACHE and (len(_SUPPORT_CACHE) >= 8 or
                _SUPPORT_CACHE_BYTES + support.cache_bytes > _SUPPORT_CACHE_LIMIT):
            _, old = _SUPPORT_CACHE.popitem(last=False)
            _SUPPORT_CACHE_BYTES -= old.cache_bytes
        _SUPPORT_CACHE[signature] = support
        _SUPPORT_CACHE_BYTES += support.cache_bytes
    return support
