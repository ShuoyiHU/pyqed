"""Reproducible H2/H4 LETTA ground-state and orbital-ordering experiments.

Run as ``python -m pyqed._letta_one_site_opt.benchmarks.qchem_ground_state``.
PySCF is optional for the library adapter but required by this benchmark.
Dense vectors, FCI and local metric audits here are small-system diagnostics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from .. import (IdentityEnvironmentCache, LETTADMROptions, canonicalize_frontier,
                letta_dmrg)
from ..qchem import ElectronicProblem, OCCUPATIONS, initial_state, embed_ties
from ..orbital_ordering import (correlation_order, graph_diagnostics,
                               orbital_mutual_information, ordering_cost,
                               select_long_range_ties, tie_neighborhoods)


GEOMETRIES = {
    'h2_equilibrium': 'H 0 0 0; H 0 0 0.74',
    'h2_stretched': 'H 0 0 0; H 0 0 2.0',
    'h4_chain': 'H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8',
    'h4_dimers': 'H 0 0 0; H 0 0 1.6; H 8 0 0; H 8 0 1.6',
}


def molecular_problem(case, orbital_basis='lowdin'):
    from pyscf import ao2mo, gto, scf
    mol = gto.M(atom=GEOMETRIES[case], basis='sto-3g', unit='Angstrom', spin=0, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('RHF failed to converge')
    if orbital_basis == 'canonical':
        coeff = mf.mo_coeff
    elif orbital_basis == 'lowdin':
        values, vectors = np.linalg.eigh(mf.get_ovlp())
        coeff = (vectors / np.sqrt(values)) @ vectors.T
    else:
        raise ValueError('orbital_basis must be canonical or lowdin')
    h1 = coeff.T @ mf.get_hcore() @ coeff
    eri = ao2mo.kernel(mol, coeff, compact=False).reshape((len(h1),)*4)
    problem = ElectronicProblem(h1, eri, mol.nelec, mol.energy_nuc())
    return problem, dict(geometry=GEOMETRIES[case], geometry_unit='Angstrom', basis='sto-3g',
                         orbital_basis=orbital_basis, rhf_total_energy=float(mf.e_tot),
                         orbital_orthogonality_error=float(np.linalg.norm(coeff.T@mf.get_ovlp()@coeff-np.eye(len(h1)))))


def fci_reference(problem):
    """Independent PySCF FCI energy and coefficients in our interleaved basis."""
    from pyscf import fci
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-13
    energy, ci = solver.kernel(problem.h1, problem.eri, problem.norb, problem.nelec,
                               ecore=problem.ecore)
    if not solver.converged:
        raise RuntimeError('FCI reference failed to converge')
    alpha = fci.cistring.make_strings(range(problem.norb), problem.nelec[0])
    beta = fci.cistring.make_strings(range(problem.norb), problem.nelec[1])
    v = np.zeros(4**problem.norb)
    for ia, a in enumerate(alpha):
        for ib, b in enumerate(beta):
            aa = [(int(a) >> i) & 1 for i in range(problem.norb)]
            bb = [(int(b) >> i) & 1 for i in range(problem.norb)]
            sign = (-1)**sum(bb[i]*aa[j] for i in range(problem.norb) for j in range(i+1, problem.norb))
            flat = sum((aa[i]+2*bb[i])*4**(problem.norb-1-i) for i in range(problem.norb))
            v[flat] = sign * ci[ia, ib]
    return float(energy), v


def metric_audit(state):
    """Audit each center with the existing contracted metric, not a dense frame."""
    centers = []
    for site in range(state.nsites):
        copy = state.copy()
        reports = canonicalize_frontier(copy, site)
        cache = IdentityEnvironmentCache(copy)
        metric = cache.effective_metric(cache.build_left_environments()[site],
                                         cache.build_right_environments()[site+1], site)
        matrix = metric.to_dense()
        if copy.symmetry is not None:
            allowed = np.flatnonzero(copy.symmetry_mask(site).ravel())
            matrix = matrix[np.ix_(allowed, allowed)]
        values = np.linalg.eigvalsh(matrix)
        supported = values[values > max(1e-12*float(values[-1]), 1e-14)]
        centers.append(dict(site=site, rank=len(supported), dimension=len(values),
                            support_condition=float(supported[-1]/supported[0]) if len(supported) else None,
                            support_identity_error=float(np.max(np.abs(supported-1))) if len(supported) else None,
                            full_identity=bool(len(supported)==len(values) and np.max(np.abs(values-1))<1e-9),
                            applied_cuts=sum(r.applied for r in reports)))
    return centers


def solve_record(problem, mpo, reference_energy, reference_vector, mps, neighborhoods,
                 *, name, max_sweeps=30, audit=True):
    state = embed_ties(mps, neighborhoods)
    start = perf_counter()
    result = letta_dmrg(mpo, state=state, options=LETTADMROptions(
        max_sweeps=max_sweeps, tolerance=1e-11, eigensolver_tolerance=1e-11,
        gauge_mode='frontier', matrix_free=True, dense_solver_threshold=64))
    elapsed = perf_counter()-start
    v = result.state.state_vector()
    v /= np.linalg.norm(v)
    # Restricted to four orbitals by this benchmark's fixed molecule presets.
    dense = mpo.to_dense()
    energy = float(np.vdot(v, dense@v).real)
    residual = float(np.linalg.norm(dense@v-energy*v))
    configs = np.array(list(np.ndindex(*((4,)*problem.norb))))
    counts = OCCUPATIONS[configs].sum(axis=1)
    probability = abs(v)**2
    sector = np.all(counts == problem.nelec, axis=1)
    kinds = Counter(u.metric_kind for sweep in result.history for u in sweep.updates)
    return dict(method=name, total_energy=energy, energy_error=energy-reference_energy,
                solver_energy_consistency=abs(energy-result.energy), residual_norm=residual,
                energy_variance=residual**2, fci_overlap=float(abs(np.vdot(reference_vector,v))**2),
                sector_leakage=float(probability[~sector].sum()),
                mean_nalpha=float(probability@counts[:,0]), mean_nbeta=float(probability@counts[:,1]),
                initial_energy=float(state.expectation(mpo)),
                parameter_count=result.state.parameter_count, stored_entries=result.state.dense_parameter_count,
                bond_dimensions=list(result.state.bond_dimensions), neighborhoods=neighborhoods,
                graph=graph_diagnostics(neighborhoods), sweep_stagnation_converged=result.converged,
                sweeps=result.sweeps, seconds=elapsed, metric_kinds=dict(kinds),
                canonical_metric_hits=result.canonical_metric_hits,
                energies=[s.energy for s in result.history],
                metric_audit=metric_audit(result.state) if audit else None)


def run_case(case, *, orbital_basis='lowdin', orders=('natural',), methods=('mps','nn'),
             max_bond_dim=9, max_sweeps=30, seed=731, audit=True):
    import pyscf, scipy
    problem, metadata = molecular_problem(case, orbital_basis)
    eref, vref = fci_reference(problem)
    entropy, weights = orbital_mutual_information(vref, problem.norb)
    optimized = correlation_order(weights)
    known = dict(natural=tuple(range(problem.norb)), correlation=optimized)
    known['scrambled'] = (0,2,3,1) if problem.norb==4 else (0,1)
    report = dict(case=case, **metadata, nelec=problem.nelec, ecore=problem.ecore,
                  fci_total_energy=eref, seed=seed, max_bond_dim=max_bond_dim,
                  max_sweeps=max_sweeps, integral_cutoff=0.,
                  versions=dict(python=platform.python_version(), numpy=np.__version__,
                                scipy=scipy.__version__, pyscf=pyscf.__version__),
                  affinity_source='FCI pilot (oracle diagnostic, not a scalable prescription)',
                  mutual_information_convention='S_i+S_j-S_ij, natural logarithms',
                  one_orbital_entropies=entropy.tolist(), mutual_information=weights.tolist(),
                  correlation_order=optimized, records=[])
    for order_name in orders:
        order = known[order_name]
        ordered = problem.reordered(order)
        mpo = ordered.mpo()
        reference, vector = fci_reference(ordered)
        dense = mpo.to_dense()
        reference_residual = float(np.linalg.norm(dense@vector-reference*vector))
        if reference_residual > 1e-8 or abs(reference-eref)>1e-9:
            raise AssertionError('MPO/FCI or orbital-permutation reference check failed')
        ordered_weights = weights[np.ix_(order, order)]
        edges = select_long_range_ties(ordered_weights)
        mps = initial_state(ordered, max_bond_dim=max_bond_dim, seed=seed)
        for method in methods:
            neighborhoods = tie_neighborhoods(problem.norb, edges if method in {'direct','carried'} else (),
                                              nearest=method!='mps', carry=method=='carried')
            if method not in {'mps','nn','direct','carried'}:
                raise ValueError('unknown method')
            row = solve_record(ordered, mpo, reference, vector, mps, neighborhoods,
                               name=method, max_sweeps=max_sweeps, audit=audit)
            row.update(order_name=order_name, orbital_order=order,
                       ordering_cost=ordering_cost(weights,order), selected_long_edges=edges,
                       mpo_bond_dimensions=mpo.bond_dimensions,
                       fci_reference_residual=reference_residual)
            report['records'].append(row)
            print(f"{case} {order_name} {method}: E={row['total_energy']:.12f} "
                  f"error={row['energy_error']:.3e} residual={row['residual_norm']:.3e}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=GEOMETRIES, default=['h2_equilibrium','h2_stretched','h4_chain'])
    parser.add_argument('--orbital-basis', choices=['lowdin','canonical'], default='lowdin')
    parser.add_argument('--orders', nargs='+', choices=['natural','scrambled','correlation'], default=['natural'])
    parser.add_argument('--methods', nargs='+', choices=['mps','nn','direct','carried'], default=['mps','nn'])
    parser.add_argument('--bond-dim', type=int, default=9)
    parser.add_argument('--max-sweeps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=731)
    parser.add_argument('--no-metric-audit', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_sweeps<1:
        parser.error('--max-sweeps must be positive')
    result = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        result.append(run_case(case, orbital_basis=args.orbital_basis, orders=args.orders,
                               methods=args.methods, max_bond_dim=args.bond_dim,
                               max_sweeps=args.max_sweeps, seed=args.seed, audit=not args.no_metric_audit))
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    main()
