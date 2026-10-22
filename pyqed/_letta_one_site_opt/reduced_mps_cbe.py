"""Full discarded-discarded residual selection for open SU(2) MPS."""
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from .reduced_symmetry import _sector_irrep


def _orth(matrix, tolerance):
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    return u[:, s > tolerance*(s[0] if s.size else 0.)]


def _discarded_blocks(blocks, left, right, tolerance):
    """Apply (1-AA†) R (1-B†B) in each weighted fusion channel."""
    projected = {}
    overlap = 0.
    for q in sorted({k[2] for k in blocks}):
        keys = [k for k in blocks if k[2] == q]
        rows = sorted({k[:2] for k in keys})
        cols = sorted({k[3:] for k in keys})
        rsh = {p: next(blocks[k].shape[:2] for k in keys if k[:2] == p) for p in rows}
        csh = {p: next(blocks[k].shape[2:] for k in keys if k[3:] == p) for p in cols}
        ro = np.cumsum([0]+[int(np.prod(rsh[p])) for p in rows])
        co = np.cumsum([0]+[int(np.prod(csh[p])) for p in cols])
        rs = {p:slice(ro[j],ro[j+1]) for j,p in enumerate(rows)}
        cs = {p:slice(co[j],co[j+1]) for j,p in enumerate(cols)}
        residual = np.zeros((ro[-1],co[-1]), dtype=np.result_type(*blocks.values()))
        for k in keys:
            residual[rs[k[:2]],cs[k[3:]]] = blocks[k].reshape(int(np.prod(rsh[k[:2]])), -1)
        da = next((v.shape[-1] for k,v in left.items() if k[-1] == q), 0)
        db = next((v.shape[0] for k,v in right.items() if k[0] == q), 0)
        a = np.zeros((ro[-1],da),dtype=residual.dtype)
        b = np.zeros((db,co[-1]),dtype=residual.dtype)
        for p in rows:
            k=(*p,q)
            if k in left:a[rs[p]] = left[k].reshape(-1,da)
        for p in cols:
            k=(q,*p)
            if k in right:
                b[:,cs[p]] = right[k].reshape(db,-1)*np.sqrt(_sector_irrep(p[-1]).dim/_sector_irrep(q).dim)
        qa, qb = _orth(a,tolerance), _orth(b.conj().T,tolerance)
        residual -= qa@(qa.conj().T@residual)
        residual -= (residual@qb)@qb.conj().T
        overlap = max(overlap, float(np.linalg.norm(qa.conj().T@residual)),
                      float(np.linalg.norm(residual@qb)))
        for k in keys:projected[k] = residual[rs[k[:2]],cs[k[3:]]].reshape(blocks[k].shape)
    return projected, overlap


def select_mps_cbe(state, hamiltonian, left_site, direction, options):
    """Exact C1/full-DD selection, adapting Gleis, Li & von Delft,
    Phys. Rev. Lett. 130, 246402 (2023), doi:10.1103/PhysRevLett.130.246402.

    Uses reduced SU(2) sector matrices and whole-multiplet SVD selection.
    This implements the paper's expensive full residual selection, not its
    shrewd preselection, hence no single-site cost claim. OBC untied MPS only.
    """
    from .reduced_coordinates import _canonical_sites
    from .reduced_state import ReducedLatticeLETTA
    from .._letta_two_site_opt.reduced_solver import (
        reduced_pair_problem, _expand_reduced_pair_space, _split_reduced_pair)
    if state.neighborhoods != tuple((i,) for i in range(state.nsites)):
        raise ValueError('mps-dd CBE requires an untied open-boundary MPS')
    i=left_site
    initial=reduced_pair_problem(state,hamiltonian,i,matrix_free=True,dense_solver_threshold=0)
    sites,_,_=_canonical_sites(initial.frontier.to_mps(state),i,last_center=i+1)
    canonical=ReducedLatticeLETTA.from_mps(sites,symmetry=state.symmetry,
        neighborhoods=state.neighborhoods)
    old=Counter(canonical.bond_sectors[i])
    scaffold=_expand_reduced_pair_space(canonical,i,sum(old.values())+options.cbe_expansion_dimension)
    problem=reduced_pair_problem(scaffold,hamiltonian,i,matrix_free=True,dense_solver_threshold=0)
    # Canonical outer environments give N_theta = dim(right irrep) I.
    # Divide the covector H theta by sqrt(dim(right)*dim(middle)) to
    # obtain the spin-weighted Schmidt matrices. The E theta term is
    # annihilated by either discarded projector and need not be subtracted.
    blocks={k:v/np.sqrt(_sector_irrep(k[-1]).dim*_sector_irrep(k[2]).dim)
            for k,v in problem.layout.unpack(problem.apply_hamiltonian(problem.old_vector)).items()}
    dd,overlap=_discarded_blocks(blocks,canonical.tensors[i],canonical.tensors[i+1],options.metric_tolerance)
    norm2=sum(_sector_irrep(k[2]).dim*float(np.vdot(v,v).real) for k,v in dd.items())
    total2=sum(_sector_irrep(k[2]).dim*float(np.vdot(v,v).real) for k,v in blocks.items())
    templates=problem.frontier.to_mps(scaffold)
    if norm2 <= options.metric_tolerance**2*max(1.,total2):
        return canonical, None, dict(missing_norm=np.sqrt(norm2),tangent_overlap=overlap)
    split=_split_reduced_pair(dd,templates[i],templates[i+1],
        bond_dim=options.cbe_expansion_dimension,sector_capacities=Counter(scaffold.bond_sectors[i]),
        direction='rl' if direction=='lr' else 'lr',cutoff=options.metric_tolerance)
    right={k:v*np.sqrt(_sector_irrep(k[0]).dim/_sector_irrep(k[-1]).dim)
           for k,v in split.right_blocks.items()}
    selection=SimpleNamespace(scaffold=scaffold,problem=problem,
        multiplicities={q:n for q,n in split.retained_multiplicities if n},
        left=split.left_blocks,right=right)
    return canonical,selection,dict(missing_norm=np.sqrt(norm2),tangent_overlap=overlap,
        captured_weight=1.-split.discarded_weight)


def mps_cbe_site(state, hamiltonian, site, direction, bond_dim, options):
    """Expand, optimize one site, then Schmidt trim (Gleis et al., 2023).

    Full-DD selection replaces the paper's shrewd selector. No ALS, partner
    energy alternation, two-site diagonalization, or LETTA baseline guard.
    The expanded solve is variational; subsequent Schmidt truncation need
    not lower energy. Report its discarded weight and post-trim residual.
    """
    from .reduced_solver import _energy, optimize_reduced_site
    from .reduced_updates import one_site_options, reduced_stationarity
    from .reduced_cbe import expand_reduced_cbe
    from .._letta_two_site_opt.solver import LETTATwoSiteOptions
    from .._letta_two_site_opt.reduced_solver import reduced_pair_problem, _optimize_untied_split
    before=_energy(state,hamiltonian,stable=True)
    i=site if direction=='lr' else site-1
    if not 0 <= i < state.nsites-1:
        return optimize_reduced_site(state,hamiltonian,site,one_site_options(options))
    canonical,selection,diagnostics=select_mps_cbe(state,hamiltonian,i,direction,options)
    candidate=canonical if selection is None else expand_reduced_cbe(canonical,selection,direction)
    update=optimize_reduced_site(candidate,hamiltonian,site,one_site_options(options))
    expanded_energy=_energy(candidate,hamiltonian,stable=True)
    trim_loss=0.
    if selection is not None:
        problem=reduced_pair_problem(candidate,hamiltonian,i,matrix_free=True,dense_solver_threshold=0)
        trim_options=LETTATwoSiteOptions(metric_tolerance=options.metric_tolerance,
            split_method='conditional-svd',energy_refinement_max_iterations=0,gauge_mode='frontier')
        trim=_optimize_untied_split(candidate,hamiltonian,problem,problem.old_vector,
            update.local_energy,update.metric_rank,update.residual_norm,direction,bond_dim,
            trim_options,acceptance_energy=np.inf)
        if not trim.accepted:raise RuntimeError('MPS Schmidt trim produced an invalid state')
        trim_loss=trim.conditional_discarded_weight
    state.tensors,state.bond_sectors=candidate.tensors,candidate.bond_sectors
    audit=reduced_stationarity(state,hamiltonian,(site,),options)[0]
    return replace(update,energy=_energy(state,hamiltonian,stable=True),cbe_selector='mps-dd',
        cbe_old_energy=before,cbe_expanded_energy=expanded_energy,cbe_trim_loss=trim_loss,
        cbe_trim_method='schmidt',cbe_projection_converged=True,
        cbe_expansion_dimension=0 if selection is None else sum(selection.multiplicities.values()),
        cbe_missing_norm=diagnostics['missing_norm'],cbe_selection_diagnostics=diagnostics,
        cbe_materialized_pair_tensor=True,cbe_materialized_pair_metric=False,
        residual_norm=audit['residual_norm'],relative_residual=audit['relative_residual'],
        local_converged=audit['relative_residual']<=options.eigensolver_tolerance)
