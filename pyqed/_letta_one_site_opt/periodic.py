"""Exact-transfer one-site optimization of periodic MPS and NN-tied LETTA.

Virtual bonds close in a trace. LETTA tensors have axes (s_i,s_{i+1},a,b),
whereas MPS tensors have axes (s_i,a,b). No open-boundary canonical metric is
assumed. The active solver contracts transfer environments, never a physical
Hilbert-space basis. Fixed charge masks enforce particle number N=L at every local update.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import linalg

from .solver import _lowest_generalized_eigenpair


NUMBER = np.array([0, 1, 1, 2])


def bond_charges(dimension):
    """Nested charge sectors, including repeated sectors for degeneracy."""
    if dimension < 1:
        raise ValueError('bond dimension must be positive')
    sequence = [0, 1, -1, 0, 1, -1]
    while len(sequence) < dimension:
        shell = (len(sequence) // 6) + 1
        sequence.extend([shell, -shell, 0, 1, -1, 0])
    return np.array(sequence[:dimension])


@dataclass
class PeriodicState:
    tensors: list
    kind: str
    charges: np.ndarray
    particle_numbers: tuple = (0, 1, 1, 2)

    def __post_init__(self):
        if self.kind not in ('mps', 'letta') or len(self.tensors) < 3:
            raise ValueError('use mps or letta on a ring with at least three sites')
        self.charges = np.asarray(self.charges, dtype=int)
        self.particle_numbers = tuple(self.particle_numbers)
        expected = (self.physical_dim,) * (2 if self.kind == 'letta' else 1) + (self.bond_dim,) * 2
        if any(a.shape != expected for a in self.tensors):
            raise ValueError('periodic tensor dimensions do not match')
        if any(np.any(a[~self.mask] != 0) for a in self.tensors):
            raise ValueError('tensors violate the unit-filling charge mask')

    @property
    def nsites(self):
        return len(self.tensors)

    @property
    def bond_dim(self):
        return len(self.charges)

    @property
    def physical_dim(self):
        return len(self.particle_numbers)

    @property
    def mask(self):
        allowed = np.asarray(self.particle_numbers)[:, None, None] + self.charges[None, :, None] - self.charges[None, None, :] == 1
        if self.kind == 'letta':
            allowed = np.broadcast_to(allowed[:, None], (self.physical_dim, self.physical_dim, self.bond_dim, self.bond_dim))
        return allowed

    def copy(self):
        return PeriodicState([a.copy() for a in self.tensors], self.kind, self.charges.copy(), self.particle_numbers)

    @classmethod
    def random(cls, nsites, bond_dim, seed=731, complex_values=False, particle_numbers=(0, 1, 1, 2)):
        rng = np.random.default_rng(seed)
        charges = bond_charges(bond_dim)
        mask = np.asarray(particle_numbers)[:, None, None] + charges[None, :, None] - charges[None, None, :] == 1
        tensors = []
        for _ in range(nsites):
            a = rng.normal(size=(len(particle_numbers), bond_dim, bond_dim))
            if complex_values:
                a = a + 1j * rng.normal(size=a.shape)
            tensors.append(a * mask / np.sqrt(bond_dim))
        result = cls(tensors, 'mps', charges, tuple(particle_numbers))
        result.normalize()
        return result

    def with_nn_ties(self):
        if self.kind != 'mps':
            raise ValueError('only an ordinary MPS needs NN ties added')
        return PeriodicState([np.repeat(a[:, None], self.physical_dim, axis=1) for a in self.tensors],
                             'letta', self.charges.copy(), self.particle_numbers)

    def padded_start(self, dimension, noise=1e-3, seed=731):
        """Initialize a larger fixed-D run; no dimension change occurs in sweeps."""
        if dimension < self.bond_dim:
            raise ValueError('padding cannot shrink a bond')
        charges = bond_charges(dimension)
        if not np.array_equal(charges[:self.bond_dim], self.charges):
            raise ValueError('padding requires nested charge sectors')
        rng = np.random.default_rng(seed)
        tensors = []
        for old in self.tensors:
            shape = old.shape[:-2] + (dimension, dimension)
            a = (rng.normal(size=shape) * noise / np.sqrt(dimension)).astype(old.dtype)
            a[..., :self.bond_dim, :self.bond_dim] = old
            mask = np.asarray(self.particle_numbers)[:,None,None] + charges[None,:,None] - charges[None,None,:] == 1
            a *= mask[:,None] if self.kind == 'letta' else mask
            tensors.append(a)
        result = PeriodicState(tensors, self.kind, charges, self.particle_numbers)
        result.normalize()
        return result

    def normalize(self):
        norm = RingContractions(self, ()).norm()
        scale = norm ** (-0.5 / self.nsites)
        self.tensors = [a * scale for a in self.tensors]

    def amplitudes(self, configurations):
        """Explicit amplitudes for tests/reference diagnostics only."""
        configurations = np.asarray(configurations, dtype=int)
        product = np.broadcast_to(np.eye(self.bond_dim),
                                  (len(configurations), self.bond_dim, self.bond_dim)).copy()
        for site, tensor in enumerate(self.tensors):
            matrices = (tensor[configurations[:, site], configurations[:, (site + 1) % self.nsites]]
                        if self.kind == 'letta' else tensor[configurations[:, site]])
            product = product @ matrices
        return np.trace(product, axis1=1, axis2=2)


def hubbard_ring_terms(nsites, hopping=1., interaction=4.):
    """Physical periodic hopping, including its complete Jordan-Wigner string."""
    from .benchmarks.condensed_models import _fermi_hubbard_terms
    if nsites < 3:
        raise ValueError('a periodic Hubbard ring needs at least three sites')
    bonds = [(i, i + 1) for i in range(nsites - 1)] + [(0, nsites - 1)]
    return _fermi_hubbard_terms(nsites, bonds,
                              dict(t=hopping, U=interaction, mu=0.))[1]


def bose_hubbard_ring_terms(nsites, hopping=1., interaction=4., max_occupancy=2):
    """Truncated bosonic hopping on a ring, with no chemical-potential shift."""
    from .benchmarks.condensed_models import _bose_hubbard_terms
    if nsites < 3:
        raise ValueError('a periodic Bose-Hubbard ring needs at least three sites')
    bonds = [(i, i + 1) for i in range(nsites - 1)] + [(0, nsites - 1)]
    return _bose_hubbard_terms(nsites, bonds,
                             dict(t=hopping, U=interaction, mu=0.,
                                  max_occupancy=max_occupancy))[1]


class RingContractions:
    """Sparse physical channels of exact cyclic double-layer transfers."""
    def __init__(self, state, terms):
        self.state = state
        self.terms = tuple(terms)
        identity = np.eye(state.physical_dim)
        self.operators = [[identity] * state.nsites]
        self.weights = [1.]
        for term in self.terms:
            self.operators.append([term.operators.get(i, identity) for i in range(state.nsites)])
            self.weights.append(term.coefficient)
        self.channels = [[(*np.nonzero(o), o[np.nonzero(o)]) for o in ops]
                         for ops in self.operators]
        self.transfers = [[self.transfer(k, i) for i in range(state.nsites)]
                          for k in range(len(self.operators))]

    def transfer(self, term, site):
        state = self.state
        a = state.tensors[site]
        if state.kind == 'mps':
            return np.einsum('pab,qcd,pq->acbd', a.conj(), a,
                             self.operators[term][site], optimize=True).reshape(
                                 state.bond_dim**2, state.bond_dim**2)
        p, q, weights = self.channels[term][site]
        r, s, _ = self.channels[term][(site + 1) % state.nsites]
        bra, ket = a[p[:, None], r[None, :]], a[q[:, None], s[None, :]]
        return np.einsum('ijab,ijcd,i->iacjbd', bra.conj(), ket, weights,
                         optimize=True).reshape(len(p) * state.bond_dim**2, len(r) * state.bond_dim**2)

    def refresh(self, sites):
        for site in set(i % self.state.nsites for i in sites):
            for k, transfers in enumerate(self.transfers):
                transfers[site] = self.transfer(k, site)

    def outside(self, term, site):
        matrices = self.transfers[term]
        order = [(site + offset) % self.state.nsites for offset in range(1, self.state.nsites)]
        product = matrices[order[0]]
        for other in order[1:]:
            product = product @ matrices[other]
        return product

    def expectation_numerator(self, term):
        matrices = self.transfers[term]
        product = matrices[0]
        for a in matrices[1:]:
            product = product @ a
        return np.trace(product)

    def norm(self):
        value = np.real_if_close(self.expectation_numerator(0))
        if np.iscomplexobj(value) or not np.isfinite(value) or value <= 0:
            raise FloatingPointError('invalid periodic state norm')
        return float(value)

    def energy(self):
        value = sum(w * self.expectation_numerator(k)
                    for k, w in enumerate(self.weights) if k) / self.norm()
        value = np.real_if_close(value)
        if np.iscomplexobj(value) or not np.isfinite(value):
            raise FloatingPointError('invalid periodic energy')
        return float(value)

    def local_term(self, term, site):
        state, d = self.state, self.state.bond_dim
        p_dim = state.physical_dim
        env = self.outside(term, site)
        if state.kind == 'mps':
            return np.einsum('pr,bdac->pabrcd', self.operators[term][site],
                             env.reshape(d, d, d, d)).reshape(p_dim*d*d, p_dim*d*d)
        p, q, weights = self.channels[term][site]
        r, s, _ = self.channels[term][(site + 1) % state.nsites]
        indices = np.arange(p_dim**2*d*d).reshape(p_dim, p_dim, d*d)
        rows, cols = indices[p[:, None], r[None, :]], indices[q[:, None], s[None, :]]
        values = env.reshape(len(r), d, d, len(p), d, d).transpose(3, 0, 4, 1, 5, 2)
        values = values.reshape(len(p), len(r), d*d, d*d) * weights[:, None, None, None]
        result = np.zeros((p_dim**2*d*d, p_dim**2*d*d), dtype=values.dtype)
        result[rows[..., :, None], cols[..., None, :]] = values
        return result

    def local_matrices(self, site):
        metric = self.local_term(0, site)
        h = sum(w * self.local_term(k, site) for k, w in enumerate(self.weights) if k)
        return (h + h.conj().T) * .5, (metric + metric.conj().T) * .5


def _charge_sqrt(gram, charges, floor):
    """Invertible, charge-preserving marginal whitening; no state truncation."""
    root, inverse = np.zeros_like(gram), np.zeros_like(gram)
    scale = max(float(np.linalg.norm(gram)), np.finfo(float).tiny)
    for charge in np.unique(charges):
        indices = np.flatnonzero(charges == charge)
        block = gram[np.ix_(indices, indices)]
        values, vectors = linalg.eigh((block + block.conj().T) * .5)
        values = np.maximum(values / scale, floor)
        root[np.ix_(indices, indices)] = (vectors * np.sqrt(values)) @ vectors.conj().T
        inverse[np.ix_(indices, indices)] = (vectors / np.sqrt(values)) @ vectors.conj().T
    return root, inverse


def balance_site(state, site, metric, floor=1e-6):
    """Whiten left/right metric marginals with exact cancelling bond gauges.

    LETTA admits separate gauges for each physical value shared on that bond.
    The residual correlated metric is still used in the generalized solve.
    """
    d = state.bond_dim
    p_dim = state.physical_dim
    tensor = metric.reshape(state.tensors[site].shape * 2)
    count = p_dim if state.kind == 'letta' else 1
    left = np.zeros((count, d, d), dtype=metric.dtype)
    right = np.zeros_like(left)
    for s in range(p_dim):
        for t in range(p_dim if state.kind == 'letta' else 1):
            block = tensor[s, t, :, :, s, t] if state.kind == 'letta' else tensor[s, :, :, s]
            left[s if state.kind == 'letta' else 0] += np.einsum('abcb->ac', block)
            right[t if state.kind == 'letta' else 0] += np.einsum('abad->bd', block)
    previous, following = (site-1) % state.nsites, (site+1) % state.nsites
    for s in range(count):
        g, inv = _charge_sqrt(left[s], state.charges, floor)
        if state.kind == 'letta':
            state.tensors[site][s] = np.einsum('ab,tbc->tac', g, state.tensors[site][s])
            state.tensors[previous][:, s] = state.tensors[previous][:, s] @ inv
        else:
            state.tensors[site] = np.einsum('ab,pbc->pac', g, state.tensors[site])
            state.tensors[previous] = state.tensors[previous] @ inv
    for t in range(count):
        g, inv = _charge_sqrt(right[t], state.charges, floor)
        if state.kind == 'letta':
            state.tensors[site][:, t] = state.tensors[site][:, t] @ g.T
            state.tensors[following][t] = np.einsum('ab,sbc->sac', inv.T, state.tensors[following][t])
        else:
            state.tensors[site] = state.tensors[site] @ g.T
            state.tensors[following] = np.einsum('ab,pbc->pac', inv.T, state.tensors[following])
    return previous, site, following


def balance_bond(state, site):
    """Invertible diagonal bond equilibration, without changing tangent spaces.

    For each virtual channel, minimize l*g**2 + r/g**2, where l and r
    are its squared factor norms. Zero channels are left alone: factoring
    a rank-deficient pair by SVD would preserve the state but can remove
    one-site search directions from its neighbor.
    """
    following = (site+1) % state.nsites
    d = state.bond_dim
    p_dim = state.physical_dim
    for physical in range(p_dim if state.kind == 'letta' else 1):
        if state.kind == 'letta':
            left = state.tensors[site][:, physical].reshape(p_dim*d, d).copy()
            right = state.tensors[following][physical].transpose(1,0,2).reshape(d,p_dim*d).copy()
        else:
            left = state.tensors[site].reshape(p_dim*d,d).copy()
            right = state.tensors[following].transpose(1,0,2).reshape(d,p_dim*d).copy()
        l = np.sum(np.abs(left)**2, axis=0)
        r = np.sum(np.abs(right)**2, axis=1)
        valid = (l > np.finfo(float).tiny) & (r > np.finfo(float).tiny)
        scales = np.ones(d)
        scales[valid] = np.exp(np.clip(.25*(np.log(r[valid])-np.log(l[valid])),
                                      np.log(1e-3), np.log(1e3)))
        left *= scales
        right /= scales[:,None]
        if state.kind == 'letta':
            state.tensors[site][:,physical] = left.reshape(p_dim,d,d)
            state.tensors[following][physical] = right.reshape(d,p_dim,d).transpose(1,0,2)
        else:
            state.tensors[site] = left.reshape(p_dim,d,d)
            state.tensors[following] = right.reshape(d,p_dim,d).transpose(1,0,2)
    return site, following


@dataclass(frozen=True)
class PeriodicOneSiteOptions:
    max_sweeps: int = 500
    energy_density_tolerance: float = 1e-11
    metric_tolerance: float = 1e-11
    energy_increase_tolerance: float = 1e-10
    gauge_floor: float = 1e-6
    gauge: bool = True
    stable_sweeps: int = 3
    gauge_method: str = 'marginal'
    paper_gauge_tolerance: float = 1e-12


def directional_gauge(state, site, direction, tolerance=1e-12):
    """Directional tensor normalization with an invertible neighboring gauge.

    In our matrix-index convention, a forward pass normalizes columns and
    transfers the inverse gauge to the next site; a backward pass normalizes
    rows. LETTA applies the same construction for each shared physical index.
    Numerically null singular directions get a finite scale, never removal:
    an exact identity is impossible on rank-deficient blocks without changing
    rank. Inverting tiny null values would needlessly amplify roundoff.
    """
    if direction not in (-1, 1):
        raise ValueError('direction must be +1 or -1')
    neighbor = (site + direction) % state.nsites
    a = state.tensors[site]
    pd, d = state.physical_dim, state.bond_dim
    tied = state.kind == 'letta'
    regularized = 0
    for physical in range(pd if tied else 1):
        block = (a[:, physical] if direction == 1 else a[physical]) if tied else a
        matrix = (block.reshape(pd*d, d) if direction == 1
                  else block.transpose(1, 0, 2).reshape(d, pd*d))
        # Charge blocks prevent roundoff mixing of distinct number sectors.
        g = np.zeros((d,d), dtype=a.dtype)
        inverse = np.zeros_like(g)
        scale = max(float(np.linalg.norm(matrix)), np.finfo(float).tiny)
        for charge in np.unique(state.charges):
            ix = np.flatnonzero(state.charges == charge)
            piece = matrix[:,ix] if direction == 1 else matrix[ix,:]
            u, singular, vh = linalg.svd(piece, full_matrices=False)
            vectors = vh.conj().T if direction == 1 else u
            floor = tolerance * scale
            regularized += int(np.count_nonzero(singular < floor))
            singular = np.where(singular < floor, scale, singular)
            g[np.ix_(ix,ix)] = (vectors*singular)@vectors.conj().T
            inverse[np.ix_(ix,ix)] = (vectors/singular)@vectors.conj().T
        if direction == 1:
            normalized = block @ inverse
            if tied:
                a[:,physical] = normalized
                state.tensors[neighbor][physical] = np.einsum(
                    'ab,tbc->tac',g,state.tensors[neighbor][physical])
            else:
                state.tensors[site] = normalized
                state.tensors[neighbor] = np.einsum('ab,sbc->sac',g,state.tensors[neighbor])
        else:
            normalized = np.einsum('ab,sbc->sac',inverse,block)
            if tied:
                a[physical] = normalized
                state.tensors[neighbor][:,physical] = state.tensors[neighbor][:,physical]@g
            else:
                state.tensors[site] = normalized
                state.tensors[neighbor] = state.tensors[neighbor]@g
    return (site, neighbor), regularized


def periodic_one_site(state, terms, options=PeriodicOneSiteOptions(), observer=None):
    """Fixed-D one-site generalized eigensolves, with no expansion/compression."""
    if options.max_sweeps < 1 or options.stable_sweeps < 1:
        raise ValueError('sweep counts must be positive')
    if options.gauge_method not in ('marginal', 'paper'):
        raise ValueError('gauge_method must be marginal or paper')
    if not 0 < options.paper_gauge_tolerance < 1:
        raise ValueError('paper gauge tolerance must lie between zero and one')
    if not (0 < options.metric_tolerance < 1 and 0 < options.gauge_floor < 1):
        raise ValueError('metric tolerance and gauge floor must lie between zero and one')
    if options.energy_density_tolerance <= 0 or options.energy_increase_tolerance < 0:
        raise ValueError('energy tolerances must be positive (increase tolerance may be zero)')
    state = state.copy()
    cache = RingContractions(state, terms)
    energy = cache.energy()
    history = [dict(sweep=0, energy=energy, elapsed_seconds=0.)]
    start, stable, converged = perf_counter(), 0, False
    retained = np.flatnonzero(state.mask.ravel())
    for sweep in range(1, options.max_sweeps + 1):
        before_sweep = energy
        rejections, gauge_rejections, ranks, residuals = 0, 0, [], []
        gauge_regularized = 0
        sites = range(state.nsites) if sweep % 2 else range(state.nsites-1, -1, -1)
        for site in sites:
            if options.gauge and options.gauge_method == 'marginal':
                old = [a.copy() for a in state.tensors]
                try:
                    changed = balance_site(state, site, cache.local_term(0, site), options.gauge_floor)
                    cache.refresh(changed)
                    check = cache.energy()
                    if abs(check-energy) > options.energy_increase_tolerance:
                        raise FloatingPointError('gauge changed energy')
                except (FloatingPointError, np.linalg.LinAlgError):
                    state.tensors = old
                    cache.refresh(range(state.nsites))
                    gauge_rejections += 1
            h, n = cache.local_matrices(site)
            h, n = h[np.ix_(retained, retained)], n[np.ix_(retained, retained)]
            candidate_energy, vector, rank, residual = _lowest_generalized_eigenpair(h, n, options.metric_tolerance)
            old = state.tensors[site].copy()
            candidate = np.zeros(old.size, dtype=np.result_type(old, vector))
            candidate[retained] = vector
            state.tensors[site] = candidate.reshape(old.shape)
            cache.refresh([site])
            try:
                check = cache.energy()
                if check > min(energy, before_sweep) + options.energy_increase_tolerance:
                    raise FloatingPointError('one-site candidate raised fresh energy')
                energy = check
            except FloatingPointError:
                state.tensors[site] = old
                cache.refresh([site])
                rejections += 1
            ranks.append(rank)
            residuals.append(residual)
            if options.gauge:
                old = [a.copy() for a in state.tensors]
                try:
                    if options.gauge_method == 'paper':
                        changed, count = directional_gauge(
                            state, site, 1 if sweep % 2 else -1, options.paper_gauge_tolerance)
                        gauge_regularized += count
                    else:
                        changed = set(balance_bond(state, (site-1) % state.nsites))
                        changed.update(balance_bond(state, site))
                    cache.refresh(changed)
                    check = cache.energy()
                    if abs(check-energy) > options.energy_increase_tolerance:
                        raise FloatingPointError('bond rebalancing changed energy')
                    energy = check
                except (FloatingPointError, np.linalg.LinAlgError):
                    state.tensors = old
                    cache.refresh(range(state.nsites))
                    gauge_rejections += 1
        # Independently rebuild transfers for acceptance and convergence.
        fresh = RingContractions(state, terms).energy()
        if abs(fresh-energy) > options.energy_increase_tolerance:
            raise FloatingPointError('cached and rebuilt periodic energies disagree')
        energy = fresh
        delta = abs(energy-before_sweep) / state.nsites
        stable = stable + 1 if delta <= options.energy_density_tolerance and not rejections else 0
        converged = stable >= options.stable_sweeps
        record = dict(sweep=sweep, energy=energy, energy_density_change=delta,
                      elapsed_seconds=perf_counter()-start, rejected_steps=rejections,
                      gauge_rejections=gauge_rejections, metric_rank_min=min(ranks),
                      gauge_regularized_values=gauge_regularized,
                      max_tensor_norm=max(float(np.linalg.norm(a)) for a in state.tensors),
                      metric_rank_max=max(ranks), max_local_residual=max(residuals), converged=converged)
        history.append(record)
        if observer is not None:
            observer(record, state)
        if converged:
            break
    return state, history, converged
