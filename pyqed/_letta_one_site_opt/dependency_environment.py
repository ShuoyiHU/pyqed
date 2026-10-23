"""Exact contraction views that remove constant dependencies of fixed cores.

The active core retains every variational parameter. Equality is exact, with
no tolerance, singular-value truncation, or reduction of the LETTA ansatz.
"""
import numpy as np
from .reduced_state import ReducedLatticeLETTA
from .reduced_frontier import ReducedFrontier
from .frontier_contraction import CanonicalBoundaryCache


def fixed_dependency_view(state, active=None):
    """Drop exactly constant tie axes from fixed tensors, without mutating state."""
    if not isinstance(state, ReducedLatticeLETTA):
        return state, ()
    neighborhoods=[];tensors=[];removed=[]
    for i,blocks in enumerate(state.tensors):
        blocks=dict(blocks);ties=list(state.neighborhoods[i][1:])
        for j in reversed(range(len(ties))):
            axis=2+j
            if i!=active and all(np.array_equal(a,np.broadcast_to(
                    np.take(a,[0],axis=axis),a.shape)) for a in blocks.values()):
                blocks={k:np.take(a,0,axis=axis) for k,a in blocks.items()}
                removed.append((i,ties.pop(j)))
        neighborhoods.append((i,*ties));tensors.append(blocks)
    if not removed:return state, ()
    view=ReducedLatticeLETTA(state.lattice_shape,state.symmetry,tensors,
        bond_sectors=state.bond_sectors,coordinates=state.coordinates,
        neighborhoods=neighborhoods,normalize=False)
    return view, tuple(removed)


class DependencySweepContext:
    """Retain local QR/metric whitening without changing other tensor gauges.

    Re-evaluate exact dependencies after every update. Boundary cache keys
    include current layouts and values, so changed dependencies invalidate reuse.
    Each context owns one transaction state; CBE candidate copies may share
    the value-validated boundary cache without sharing mutable state.
    """
    def __init__(self,state,hamiltonian, *, structured=True,boundary_cache=None):
        from .reduced_solver import StableEnergyCache
        self.structured=structured
        self.state=state;self.hamiltonian=hamiltonian
        self.boundary_cache=CanonicalBoundaryCache() if boundary_cache is None else boundary_cache
        self.energy_cache=StableEnergyCache(hamiltonian)
        self.dependency_stats={};self.energy_center=None
        self.frontier=None
        self.search_cache={}

    def stable_energy(self,state=None):
        return self.energy_cache(self.state if state is None else state,
            center=self.energy_center,canonical_boundaries=self.boundary_cache,
            structured=self.structured)

    def synchronize(self,indices):
        # Views are rebuilt from current tensors; no stale expanded cores exist.
        self.energy_cache.key=None

    def local_problem(self,site,options):
        self.energy_center=int(site)
        from .reduced_solver import _native_local_problem
        view,removed=fixed_dependency_view(self.state,active=site)
        if (self.frontier is None or self.frontier.neighborhoods != view.neighborhoods
                or self.frontier.reduced_dim != view.physical_dim):
            self.frontier=ReducedFrontier.from_state(view)
        frontier=self.frontier
        embedding=frontier.site_embedding(view,site)
        sites=frontier.to_mps(view,compact=True)
        self.dependency_stats=dict(active_site=int(site),removed_dependencies=removed,
            max_pending_labels=max(map(len,frontier.cuts),default=0),
            max_frontier_multiplets=max((len(a.qns[2]) for a in sites[:-1]),default=1),
            source_parameters=embedding.source_size,
            full_source_parameters=embedding.source_layout.size)
        problem=_native_local_problem(view,self.hamiltonian,site,sites,embedding,
            options.matrix_free,options.dense_solver_threshold,self.boundary_cache,structured=self.structured)
        import os
        if os.environ.get('PYQED_DAVIDSON_RECYCLE','0')=='1':
            support=embedding.supported_source_indices
            key=(site,tuple(view.site_neighborhood(site)),tuple(embedding.source_layout.shapes.items()),
                 None if support is None else support.tobytes())
            problem.coordinates.search_cache=(self.search_cache,key)
        return problem
