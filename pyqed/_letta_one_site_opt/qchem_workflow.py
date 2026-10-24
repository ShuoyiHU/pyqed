"""Checkpointed singlet CAS workflow using the existing reduced SU(2) solvers."""
from collections import Counter
import hashlib
from operator import index
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np

from . import ReducedLatticeLETTA, ReducedFrontier, LETTADMROptions
from .qchem import ElectronicProblem
from .orbital_information import orbital_information, strongest_disjoint, negative_conditional_ties
from .orbital_ordering import tie_neighborhoods, graph_diagnostics
from .reduced_contraction import expand_reduced_mps_site
from .reduced_operators import ReducedMPOHamiltonian
from .reduced_solver import reduced_letta_dmrg


def parameter_counts(state):
    """Stored reduced coefficients, not gauge-independent real degrees of freedom."""
    return dict(by_site=[sum(a.size for a in t.values()) for t in state.tensors],
                reduced_coefficients=sum(a.size for t in state.tensors for a in t.values()),
                multiplet_dimensions=list(state.bond_dimensions),
                magnetic_dimensions=list(state.magnetic_bond_dimensions),
                ordinary_bond_gauge_dimension=sum(sum(m*m for m in Counter(b).values()) for b in state.bond_sectors))


def compact_hamiltonian(hamiltonian, basis):
    """Drop construction-only storage while retaining the identical compiled MPO.

    No additional operator truncation is performed. This reduces retained/cache
    storage, not the original builder's peak memory. Component factors remain
    available for reference contractions.
    """
    mpo=hamiltonian.native_mpo(basis)
    from .qchem_mpo_cache import ComponentReference
    compact=ReducedMPOHamiltonian(None,tuple(ComponentReference(mpo,i) for i in range(len(mpo.sites))),name=hamiltonian.name,
                                  contraction_backend=hamiltonian.contraction_backend)
    object.__setattr__(compact,'_native_mpo_cache',{basis:mpo})
    return compact


def _signature(problem):
    digest=hashlib.sha256()
    for a in (problem.h1,problem.eri,np.array(problem.nelec),np.array(problem.orbital_order),np.array(problem.ecore)):
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


class QCLETTA:
    """Two-site MPS -> optional MPS polish -> MI ties -> one-site LETTA.

    This is the experiment's workflow adaptation, not a new solver or an
    orbital-optimization method. Real spatial CAS integrals, even-N singlets,
    fixed orbital order, and direct forward ties are supported. Tie selection
    is an entropy-based heuristic, not a guarantee of energy improvement; see Legeza
    and Solyom, PRB 68, 195116 (2003), doi:10.1103/PhysRevB.68.195116.
    Existing pyqed reduced solvers perform all variational operations. Local
    magnetic tensors are expanded only for the orbital-information diagnostic.
    ``bond_dim`` caps total SU(2) multiplets per source bond, not per sector.
    """
    def __init__(self, problem, bond_dim=30, *, seed=71, checkpoint=None):
        self.bond_dim=index(bond_dim)
        if self.bond_dim<1 or problem.nelec[0]!=problem.nelec[1]:
            raise ValueError('positive bond_dim and a singlet electron sector are required')
        self.problem=problem
        self.seed=index(seed)
        self.checkpoint=None if checkpoint is None else Path(checkpoint)
        self.state=ReducedLatticeLETTA.random((1,problem.norb),symmetry=problem.symmetry('su2'),
            neighborhoods=tuple((i,) for i in range(problem.norb)),multiplets_per_sector=1,seed=seed,real=True)
        electronic=ElectronicProblem(problem.h1,problem.eri,problem.nelec,0.,problem.orbital_order)
        tick=perf_counter()
        self.hamiltonian=compact_hamiltonian(electronic.su2_mpo(),self.state.physical_basis)
        self.preparation_seconds=perf_counter()-tick
        self.history=[]
        self.stages={}
        self.edges=[]
        self.information=None
        self.energy=None
        self.success=False
        self.message='Not optimized'
        self._last_direction={}

    def save(self):
        """Atomically save state and matching progress; load only trusted pickles."""
        if self.checkpoint is None:return
        self.checkpoint.parent.mkdir(parents=True,exist_ok=True)
        payload={k:getattr(self,k) for k in ('state','bond_dim','seed','history','stages','edges',
                 'information','energy','success','message','_last_direction','preparation_seconds')}
        payload['integral_signature']=_signature(self.problem)
        temporary=self.checkpoint.with_suffix(self.checkpoint.suffix+'.tmp')
        with temporary.open('wb') as stream:pickle.dump(payload,stream,pickle.HIGHEST_PROTOCOL)
        temporary.replace(self.checkpoint)

    @classmethod
    def load(cls, problem, checkpoint):
        """Restore a trusted checkpoint, checking the CAS integrals and order."""
        with Path(checkpoint).open('rb') as stream:payload=pickle.load(stream)
        if payload.pop('integral_signature')!=_signature(problem):
            raise ValueError('checkpoint integrals, electron sector or orbital order differ')
        obj=cls(problem,payload['bond_dim'],seed=payload['seed'],checkpoint=checkpoint)
        for name,value in payload.items():setattr(obj,name,value)
        return obj

    def _record(self, stage, state, sweep, seconds):
        self.state=state
        self.energy=float(sweep.energy+self.problem.ecore)
        self._last_direction[stage]=sweep.direction
        row=dict(stage=stage,sweep=1+sum(r['stage']==stage for r in self.history),
                 direction=sweep.direction,electronic_energy=float(sweep.energy),
                 total_energy=self.energy,seconds=seconds,**parameter_counts(state))
        self.history.append(row)
        self.save()

    def two_site(self, sweeps=39):
        """Run a requested number of additional directional sweeps, not a convergence claim."""
        from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg
        if self.edges:raise ValueError('two_site preparation requires an untied MPS')
        sweeps=index(sweeps)
        if sweeps<1:raise ValueError('sweeps must be positive')
        self.success=False;self.message='Two-site fixed-sweep preparation'
        for _ in range(sweeps):
            direction='rl' if self._last_direction.get('two_site')=='lr' else 'lr'
            tick=perf_counter()
            result=letta_two_site_dmrg(self.hamiltonian,state=self.state,bond_dim=self.bond_dim,
                options=LETTATwoSiteOptions(max_sweeps=1,start_direction=direction,tolerance=1e-11,
                    eigensolver_tolerance=1e-11,metric_tolerance=1e-12,split_method='conditional-svd',
                    reduced_sector_growth=True,energy_increase_tolerance=float('inf'),dense_solver_threshold=32))
            self._record('two_site',result.state,result.history[-1],perf_counter()-tick)
        self.stages['two_site']=dict(converged=False,message=self.message)
        self.save()
        return self

    def _one_site(self, stage, max_sweeps):
        if index(max_sweeps)<1:raise ValueError('max_sweeps must be positive')
        self.success=False;self.message=f'{stage} running'
        previous=perf_counter()
        sectors=self.state.bond_sectors
        def observe(state,sweep):
            nonlocal previous
            now=perf_counter()
            if state.bond_sectors!=sectors:raise RuntimeError('one-site solve changed source sectors')
            self._record(stage,state,sweep,now-previous)
            previous=perf_counter()  # Exclude checkpoint I/O from sweep timing.
        direction='rl' if self._last_direction.get(stage)=='lr' else 'lr'
        result=reduced_letta_dmrg(self.hamiltonian,state=self.state,
            options=LETTADMROptions(max_sweeps=max_sweeps,tolerance=1e-11,eigensolver_tolerance=1e-11,
                metric_tolerance=1e-12,energy_increase_tolerance=1e-11,gauge_mode='frontier',
                dense_solver_threshold=32,start_direction=direction),sweep_callback=observe)
        self.state=result.state;self.energy=float(result.energy+self.problem.ecore)
        self.success=bool(result.converged);self.message=result.message
        self.stages[stage]=dict(converged=self.success,message=self.message)
        self.save()
        return self

    def polish(self, max_sweeps=200):
        if self.edges:raise ValueError('polish requires an untied MPS')
        return self._one_site('mps_polish',max_sweeps)

    def tie(self, edges=None, *, selection="conditional", count=2):
        """Embed the current untied MPS exactly; edges use zero-based chain positions.

        Default: every pair with negative conditional entropy in either direction.
        ``selection="mi"`` chooses up to ``count`` disjoint positive-MI pairs.
        Explicit edges override selection. Source sectors and multiplicities
        are preserved. No Fiedler permutation or carried dependencies is implied.
        """
        if self.edges:raise ValueError('tie must start from an untied MPS')
        if max(self.state.bond_dimensions,default=1)>self.bond_dim:
            raise ValueError('prepare the MPS to enforce the requested bond cap before tying')
        sites=ReducedFrontier.from_state(self.state).to_mps(self.state)
        single,mi,work=orbital_information([expand_reduced_mps_site(s) for s in sites])
        if edges is None:
            if selection=='conditional':edges=negative_conditional_ties(single,mi)
            elif selection=='mi':edges=strongest_disjoint(mi,count)[0]
            else:raise ValueError('selection must be conditional or mi')
        else:selection='explicit'
        neighborhoods=tie_neighborhoods(self.problem.norb,edges=edges)
        self.edges=[(i,j) for i,neighbors in enumerate(neighborhoods) for j in neighbors[1:]]
        self.information=dict(single_entropy=single.tolist(),mutual_information=mi.tolist(),
            selection=selection,work=work,graph=graph_diagnostics(neighborhoods,physical_dim=3))
        self.state=ReducedLatticeLETTA.from_mps(sites,symmetry=self.problem.symmetry('su2'),neighborhoods=neighborhoods)
        self.success=False;self.message='Exact MPS embedding; LETTA not optimized'
        self.save()
        return self

    def optimize_letta(self, max_sweeps=200):
        return self._one_site('letta',max_sweeps)

    def report(self):
        return dict(bond_dim=self.bond_dim,seed=self.seed,integral_signature=_signature(self.problem),
                    energy=self.energy,success=self.success,message=self.message,stages=self.stages,
                    edges=self.edges,information=self.information,history=self.history,
                    preparation_seconds=self.preparation_seconds,parameters=parameter_counts(self.state))
