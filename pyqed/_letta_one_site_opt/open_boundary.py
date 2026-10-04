"""Open virtual MPS/NN LETTA, including optional last-first physical tie.

Hamiltonian product terms may be periodic. Only double-layer transfer networks
are used in optimization; explicit amplitudes are reference helpers.
"""
from collections import Counter
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import linalg

from .periodic import RingContractions, bond_charges
from .solver import _lowest_generalized_eigenpair


def open_bond_charges(length, dimension, numbers):
    """Cap sector multiplicities by both physical prefix and suffix capacities."""
    counts = [Counter({0: 1})]
    for _ in range(length):
        new = Counter()
        for q, count in counts[-1].items():
            for n in numbers:
                new[q + n - 1] += count
        counts.append(new)
    bonds = [np.array([0])]
    for cut in range(1, length):
        selected, used = [], Counter()
        for q in bond_charges(dimension):
            if used[q] < min(counts[cut][q], counts[length-cut][-q]):
                selected.append(q)
                used[q] += 1
        bonds.append(np.array(selected, dtype=int))
    return bonds + [np.array([0])]


@dataclass
class OpenState:
    tensors: list
    kind: str
    charges: list
    particle_numbers: tuple = (0, 1, 1, 2)
    wrap_tie: bool = False

    def __post_init__(self):
        self.charges = [np.asarray(q, dtype=int) for q in self.charges]
        if self.kind not in ('mps', 'letta') or len(self.tensors) < 3:
            raise ValueError('expected MPS or LETTA with at least three sites')
        if self.kind == 'mps' and self.wrap_tie:
            raise ValueError('MPS has no physical-index ties')
        if len(self.charges) != self.nsites + 1:
            raise ValueError('one charge list is required per virtual bond')
        if any(not np.array_equal(q, [0]) for q in (self.charges[0], self.charges[-1])):
            raise ValueError('open endpoints must have dimension one and charge zero')
        for i, a in enumerate(self.tensors):
            if a.shape != self.mask(i).shape or np.any(a[~self.mask(i)] != 0):
                raise ValueError(f'invalid tensor shape or charge mask at site {i}')

    @property
    def nsites(self):
        return len(self.tensors)

    @property
    def physical_dim(self):
        return len(self.particle_numbers)

    def tied(self, site):
        return self.kind == 'letta' and (site < self.nsites - 1 or self.wrap_tie)

    def mask(self, site):
        dl, dr = len(self.charges[site]), len(self.charges[site+1])
        allowed = (np.asarray(self.particle_numbers)[:, None, None]
                   + self.charges[site][None, :, None]
                   - self.charges[site+1][None, None, :] == 1)
        if self.tied(site):
            allowed = np.broadcast_to(allowed[:, None], (self.physical_dim, self.physical_dim, dl, dr))
        return allowed

    def copy(self):
        return OpenState([a.copy() for a in self.tensors], self.kind,
                         [q.copy() for q in self.charges], self.particle_numbers, self.wrap_tie)

    @classmethod
    def random(cls, length, dimension, seed=731, particle_numbers=(0, 1, 1, 2), complex_values=False):
        rng = np.random.default_rng(seed)
        charges = open_bond_charges(length, dimension, particle_numbers)
        tensors = []
        for i in range(length):
            shape = (len(particle_numbers), len(charges[i]), len(charges[i+1]))
            a = rng.normal(size=shape)
            if complex_values:
                a = a + 1j*rng.normal(size=shape)
            mask = (np.asarray(particle_numbers)[:,None,None] + charges[i][None,:,None]
                    - charges[i+1][None,None,:] == 1)
            tensors.append(a*mask/np.sqrt(dimension))
        state = cls(tensors, 'mps', charges, tuple(particle_numbers))
        state.tensors[0] /= np.sqrt(OpenContractions(state, ()).norm())
        return state

    def with_nn_ties(self, wrap=False):
        if self.kind != 'mps':
            raise ValueError('start from MPS to add ties')
        tensors = [np.repeat(a[:,None], self.physical_dim, axis=1)
                   if i < self.nsites-1 or wrap else a.copy()
                   for i,a in enumerate(self.tensors)]
        return OpenState(tensors, 'letta', self.charges, self.particle_numbers, wrap)

    def amplitudes(self, configurations):
        """Reference only; never called by the active optimizer."""
        c = np.asarray(configurations, dtype=int)
        product = np.ones((len(c), 1, 1))
        for i,a in enumerate(self.tensors):
            block = a[c[:,i], c[:,(i+1)%self.nsites]] if self.tied(i) else a[c[:,i]]
            product = product @ block
        return product[:,0,0]


class OpenContractions(RingContractions):
    """Exact product-operator contractions with variable virtual dimensions.

    Cyclic physical-channel bookkeeping is also used for an untied last tensor:
    it broadcasts over the first physical channel, with no last-first coupling.
    The virtual endpoint transfer dimension is one for MPS.
    """
    def transfer(self, term, site):
        state = self.state
        a = state.tensors[site]
        dl, dr = a.shape[-2:]
        if state.kind == 'mps':
            return np.einsum('pab,qcd,pq->acbd', a.conj(), a,
                             self.operators[term][site], optimize=True).reshape(dl*dl, dr*dr)
        p,q,w = self.channels[term][site]
        r,s,_ = self.channels[term][(site+1)%state.nsites]
        if state.tied(site):
            bra,ket = a[p[:,None],r[None,:]], a[q[:,None],s[None,:]]
        else:
            bra = np.broadcast_to(a[p,None], (len(p),len(r),dl,dr))
            ket = np.broadcast_to(a[q,None], (len(p),len(r),dl,dr))
        return np.einsum('ijab,ijcd,i->iacjbd',bra.conj(),ket,w,optimize=True).reshape(
            len(p)*dl*dl,len(r)*dr*dr)

    def local_term(self, term, site):
        state = self.state
        dl,dr = state.tensors[site].shape[-2:]
        pd = state.physical_dim
        env = self.outside(term,site)
        if state.kind == 'mps':
            return np.einsum('pr,bdac->pabrcd',self.operators[term][site],
                             env.reshape(dr,dr,dl,dl)).reshape(pd*dl*dr,pd*dl*dr)
        p,q,w = self.channels[term][site]
        r,s,_ = self.channels[term][(site+1)%state.nsites]
        env = env.reshape(len(r),dr,dr,len(p),dl,dl)
        if state.tied(site):
            indices = np.arange(pd*pd*dl*dr).reshape(pd,pd,dl*dr)
            rows,cols = indices[p[:,None],r[None,:]], indices[q[:,None],s[None,:]]
            values = env.transpose(3,0,4,1,5,2).reshape(len(p),len(r),dl*dr,dl*dr)
            values = values*w[:,None,None,None]
        else:
            indices = np.arange(pd*dl*dr).reshape(pd,dl*dr)
            rows,cols = indices[p],indices[q]
            values = env.sum(axis=0).transpose(2,3,0,4,1).reshape(len(p),dl*dr,dl*dr)
            values = values*w[:,None,None]
        size = state.tensors[site].size
        result = np.zeros((size,size),dtype=values.dtype)
        result[rows[..., :,None],cols[...,None,:]] = values
        return result


def _supported_root(gram, charges, tolerance):
    root, inverse = np.zeros_like(gram), np.zeros_like(gram)
    scale = max(float(np.linalg.norm(gram, 2)), np.finfo(float).tiny)
    for q in np.unique(charges):
        ix = np.flatnonzero(charges == q)
        block = gram[np.ix_(ix,ix)]
        values, vectors = linalg.eigh((block+block.conj().T)*.5)
        if values.min() < -1e-10*scale:
            raise FloatingPointError('norm marginal is not positive semidefinite')
        # An invertible gauge preserves even null one-site parameter directions.
        values = np.where(values > tolerance*scale, values, scale)
        root[np.ix_(ix,ix)] = (vectors*np.sqrt(values))@vectors.conj().T
        inverse[np.ix_(ix,ix)] = (vectors/np.sqrt(values))@vectors.conj().T
    return root,inverse


def whiten_open_site(state, site, metric, tolerance=1e-12):
    """Whiten charge-block norm marginals with cancelling neighbor gauges.

    For open MPS this produces a scalar identity on the supported norm space.
    LETTA gauges may depend on the physical index shared across a bond. With a
    wrap tie the environment can remain correlated, so the full metric is kept.
    """
    a = state.tensors[site]
    dl,dr = a.shape[-2:]
    pd = state.physical_dim
    tied_right = state.tied(site)
    tied_left = state.tied((site-1)%state.nsites)
    tensor = metric.reshape(a.shape*2)
    left = np.zeros((pd if tied_left else 1,dl,dl),dtype=metric.dtype)
    right = np.zeros((pd if tied_right else 1,dr,dr),dtype=metric.dtype)
    for s in range(pd):
        for t in range(pd if tied_right else 1):
            block = tensor[s,t,:,:,s,t] if tied_right else tensor[s,:,:,s]
            left[s if tied_left else 0] += np.einsum('abcb->ac',block)
            right[t if tied_right else 0] += np.einsum('abad->bd',block)
    previous,following = (site-1)%state.nsites,(site+1)%state.nsites
    changed = {site}
    if site > 0 or tied_left:
        for s in range(len(left)):
            g,inv = _supported_root(left[s],state.charges[site],tolerance)
            if tied_left:
                a[s] = np.einsum('ab,...bc->...ac',g,a[s])
                state.tensors[previous][:,s] = state.tensors[previous][:,s]@inv
            else:
                state.tensors[site] = a = np.einsum('ab,...bc->...ac',g,a)
                state.tensors[previous] = state.tensors[previous]@inv
        changed.add(previous)
    if site < state.nsites-1 or tied_right:
        for t in range(len(right)):
            g,inv = _supported_root(right[t],state.charges[site+1],tolerance)
            if tied_right:
                a[:,t] = a[:,t]@g.T
                state.tensors[following][t] = np.einsum('ab,...bc->...ac',inv.T,state.tensors[following][t])
            else:
                state.tensors[site] = a = a@g.T
                state.tensors[following] = np.einsum('ab,...bc->...ac',inv.T,state.tensors[following])
        changed.add(following)
    return changed


def balance_tensor_scales(state):
    """Cancelling scalar bond gauges keep tensor norms at their geometric mean."""
    logs = np.log([np.linalg.norm(a) for a in state.tensors])
    if not np.all(np.isfinite(logs)):
        raise FloatingPointError('nonfinite tensor norm')
    shifts = logs.mean() - logs
    # Enforce a unit product of scales despite the rounding of the mean.
    shifts[-1] = -sum(shifts[:-1])
    state.tensors = [a*np.exp(shift) for a,shift in zip(state.tensors,shifts)]


@dataclass(frozen=True)
class OpenOneSiteOptions:
    max_sweeps: int = 500
    energy_density_tolerance: float = 1e-11
    metric_tolerance: float = 1e-11
    energy_increase_tolerance: float = 1e-10
    gauge_tolerance: float = 1e-12
    stable_sweeps: int = 3


def open_one_site(initial, terms, options=OpenOneSiteOptions(), observer=None):
    """Fixed-bond one-site solves; no expansion or physical-basis projection."""
    if options.max_sweeps < 1 or options.stable_sweeps < 1:
        raise ValueError('sweep counts must be positive')
    if not 0 < options.metric_tolerance < 1 or not 0 < options.gauge_tolerance < 1:
        raise ValueError('metric and gauge tolerances must be between zero and one')
    if options.energy_density_tolerance <= 0 or options.energy_increase_tolerance < 0:
        raise ValueError('invalid energy tolerance')
    state = initial.copy()
    cache = OpenContractions(state,terms)
    energy = cache.energy()
    history = [dict(sweep=0,energy=energy,elapsed_seconds=0.)]
    start,stable,converged = perf_counter(),0,False
    for sweep in range(1,options.max_sweeps+1):
        before = energy
        rejections,gauge_rejections = 0,0
        residuals,ranks,conditions = [],[],[]
        sites = range(state.nsites) if sweep%2 else range(state.nsites-1,-1,-1)
        for site in sites:
            old = [a.copy() for a in state.tensors]
            try:
                whiten_open_site(state,site,cache.local_term(0,site),options.gauge_tolerance)
                balance_tensor_scales(state)
                cache.refresh(range(state.nsites))
                check = cache.energy()
                if abs(check-energy) > options.energy_increase_tolerance:
                    raise FloatingPointError('gauge changed energy')
                energy = check
            except (FloatingPointError,np.linalg.LinAlgError):
                state.tensors = old
                cache.refresh(range(state.nsites))
                gauge_rejections += 1
            retained = np.flatnonzero(state.mask(site).ravel())
            h,n = cache.local_matrices(site)
            h,n = h[np.ix_(retained,retained)],n[np.ix_(retained,retained)]
            diagonal = np.real(np.diag(n))
            positive = diagonal > np.finfo(float).tiny
            eq = n[np.ix_(positive,positive)]/np.sqrt(diagonal[positive,None]*diagonal[None,positive])
            values = linalg.eigvalsh(eq)
            support = values[values > options.metric_tolerance*values[-1]]
            conditions.append(float(support[-1]/support[0]))
            _,vector,rank,residual = _lowest_generalized_eigenpair(h,n,options.metric_tolerance)
            old = state.tensors[site].copy()
            candidate = np.zeros(old.size,dtype=np.result_type(old,vector))
            candidate[retained] = vector
            state.tensors[site] = candidate.reshape(old.shape)
            cache.refresh([site])
            try:
                check = cache.energy()
                if check > energy + options.energy_increase_tolerance:
                    raise FloatingPointError('one-site update increased energy')
                energy = check
            except FloatingPointError:
                state.tensors[site] = old
                cache.refresh([site])
                rejections += 1
            ranks.append(rank)
            residuals.append(residual)
        fresh = OpenContractions(state,terms).energy()
        if abs(fresh-energy) > options.energy_increase_tolerance:
            raise FloatingPointError('rebuilt and cached energies disagree')
        energy = fresh
        delta = abs(energy-before)/state.nsites
        stable = stable+1 if delta <= options.energy_density_tolerance and not rejections else 0
        converged = stable >= options.stable_sweeps
        row = dict(sweep=sweep,energy=energy,elapsed_seconds=perf_counter()-start,
                   energy_density_change=delta,rejected_steps=rejections,gauge_rejections=gauge_rejections,
                   max_local_residual=max(residuals),metric_rank_min=min(ranks),metric_rank_max=max(ranks),
                   max_equilibrated_metric_condition=max(conditions),
                   max_tensor_norm=max(float(np.linalg.norm(a)) for a in state.tensors),converged=converged)
        history.append(row)
        if observer is not None:
            observer(row,state)
        if converged:
            break
    return state,history,converged
