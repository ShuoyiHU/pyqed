"""Spin-purity, adaptive-MPS, and exact NN-embedding validation.

Run with PYTHONPATH=. and single-threaded BLAS. PySCF is used only for the
independent full-CI reference; production optimization uses MPO environments.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
from collections import Counter

import numpy as np
from pyscf import fci

from pyqed._letta_one_site_opt import (
    ReducedLatticeLETTA, ReducedFrontier, LETTADMROptions, letta_dmrg,
)
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.orbital_ordering import tie_neighborhoods
from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg


def spin_residual(state):
    """Explicit small-system oracle, used only after optimization."""
    n = state.nsites
    v = state.state_vector()
    plus = np.zeros((4, 4)); plus[1, 2] = 1.
    z = np.diag([0., .5, -.5, 0.])
    def total_action(local, vector):
        tensor = np.asarray(vector).reshape((4,)*n)
        result = np.zeros_like(tensor, dtype=np.result_type(tensor, local))
        for i in range(n):
            result += np.moveaxis(np.tensordot(local, tensor, axes=(1, i)), 0, i)
        return result.reshape(-1)
    action = total_action(z, total_action(z, v)) + .5*(
        total_action(plus, total_action(plus.T, v))
        + total_action(plus.T, total_action(plus, v)))
    S = state.target_two_j/2
    return float(np.linalg.norm(action-S*(S+1)*v)/np.linalg.norm(v))


def run():
    records = []
    for n, nelec, two_s, cap in [(3, (2, 1), 1, 5), (4, (2, 2), 0, 8)]:
        h = -np.eye(n, k=1)-np.eye(n, k=-1)
        g = np.zeros((n,)*4)
        for i in range(n):
            g[i, i, i, i] = 4.
        problem = ElectronicProblem(h, g, nelec)
        mpo = problem.su2_mpo()
        symmetry = problem.symmetry('su2', two_s=two_s)
        initial = ReducedLatticeLETTA.random((1, n), symmetry=symmetry,
            neighborhoods=tuple((i,) for i in range(n)), seed=56)
        mps = letta_two_site_dmrg(mpo, state=initial, bond_dim=cap,
            options=LETTATwoSiteOptions(max_sweeps=4, tolerance=1e-10,
                split_method='conditional-svd', reduced_sector_growth=True,
                gauge_mode='frontier', truncation_max_iterations=12))
        tied = ReducedLatticeLETTA.from_mps(
            ReducedFrontier.from_state(mps.state).to_mps(mps.state),
            symmetry=symmetry, neighborhoods=tie_neighborhoods(n, nearest=True))
        embedding_distance = float(np.linalg.norm(tied.state_vector()-mps.state.state_vector()))
        result = letta_dmrg(mpo, state=tied, options=LETTADMROptions(
            max_sweeps=4, tolerance=1e-11, gauge_mode='frontier'))
        reference, _ = fci.direct_spin1.kernel(h, g, n, nelec, verbose=0)
        residual = spin_residual(result.state)
        if embedding_distance > 1e-12 or residual > 1e-10:
            raise AssertionError('embedding or spin-purity check failed')
        if result.energy > mps.energy+1e-9 or result.energy < reference-1e-9:
            raise AssertionError('variational energy check failed')
        row = dict(model='open Hubbard chain, t=1, U=4', nsites=n,
            nelec=nelec, two_s=two_s, multiplet_cap=cap,
            multiplet_dimensions=result.state.bond_dimensions,
            magnetic_dimensions=result.state.magnetic_bond_dimensions,
            untied_energy=mps.energy, nn_energy=result.energy,
            fci_energy=float(reference), embedding_distance=embedding_distance,
            spin_residual=residual, component_norm=result.state.norm())
        records.append(row)
        print(json.dumps(row), flush=True)
    return records


def run_molecular(path, cap, sweeps):
    """Adaptive two-site MPS control and its exact NN LETTA embedding."""
    if cap < 1 or sweeps < 1:
        raise ValueError('cap and sweeps must be positive')
    data = np.load(path)
    problem = ElectronicProblem(data['h1'], data['eri'], tuple(data['nelec']),
                                float(data['ecore']), tuple(data['orbital_order']))
    n = problem.norb
    sym = problem.symmetry('su2')
    start = perf_counter()
    mpo = problem.su2_mpo()
    mpo.native_mpo(sym.physical_basis)
    setup_seconds = perf_counter()-start
    reference, ci = fci.direct_spin1.kernel(problem.h1, problem.eri, n,
        problem.nelec, ecore=problem.ecore, verbose=0)
    target_spin = sym.sector.irrep.two_j/2
    if abs(fci.spin_op.spin_square(ci, n, problem.nelec)[0]
           - target_spin*(target_spin+1)) > 1e-8:
        raise ValueError('FCI reference does not have the requested total spin')
    initial = ReducedLatticeLETTA.random((1, n), symmetry=sym,
        neighborhoods=tuple((i,) for i in range(n)), multiplets_per_sector=1, seed=71)
    start = perf_counter()
    mps = letta_two_site_dmrg(mpo, state=initial, bond_dim=cap,
        options=LETTATwoSiteOptions(max_sweeps=sweeps, tolerance=1e-11,
            split_method='conditional-svd', reduced_sector_growth=True,
            gauge_mode='frontier', truncation_max_iterations=12,
            dense_solver_threshold=32, verbosity=1))
    mps_seconds = perf_counter()-start
    tied = ReducedLatticeLETTA.from_mps(ReducedFrontier.from_state(mps.state).to_mps(mps.state),
        symmetry=sym, neighborhoods=tie_neighborhoods(n, nearest=True))
    embedding_distance = float(np.linalg.norm(tied.state_vector()-mps.state.state_vector()))
    start = perf_counter()
    result = letta_dmrg(mpo, state=tied, options=LETTADMROptions(
        max_sweeps=sweeps, tolerance=1e-11, gauge_mode='frontier',
        dense_solver_threshold=32, verbosity=1))
    nn_seconds = perf_counter()-start
    residual, mps_residual = spin_residual(result.state), spin_residual(mps.state)
    if embedding_distance > 1e-11 or max(residual, mps_residual) > 1e-9:
        raise AssertionError('embedding or spin-purity check failed')
    if result.energy > mps.energy+1e-9 or min(result.energy, mps.energy) < reference-1e-8:
        raise AssertionError('variational/inclusion energy check failed')
    if max(mps.state.bond_dimensions) > cap:
        raise AssertionError('adaptive MPS exceeded its multiplet cap')
    if mps.state.bond_sectors != result.state.bond_sectors:
        raise AssertionError('NN refinement changed the MPS sector allocation')
    norm = result.state.norm()
    if abs(norm-1.) > 1e-10:
        raise AssertionError('NN refinement is not normalized')
    return dict(model=path.stem, nsites=n, nelec=problem.nelec,
        two_s=sym.sector.irrep.two_j, seed=71, orbital_order=problem.orbital_order,
        multiplet_cap=cap, mps_energy=mps.energy, nn_energy=result.energy,
        fci_energy=float(reference), mps_sweeps=mps.sweeps, nn_sweeps=result.sweeps,
        setup_seconds=setup_seconds, mps_seconds=mps_seconds, nn_refinement_seconds=nn_seconds,
        multiplet_dimensions=result.state.bond_dimensions,
        magnetic_dimensions=result.state.magnetic_bond_dimensions,
        nn_frontiers=ReducedFrontier.from_state(result.state).cuts,
        adaptive_sector_multiplicities=[[(q.charge, q.irrep.two_j, r) for q, r in Counter(bond).items()]
                                       for bond in mps.state.bond_sectors],
        embedding_distance=embedding_distance, spin_residual=residual,
        mps_spin_residual=mps_residual, component_norm=norm,
        mps_history=[dict(energy=x.energy, accepted_updates=sum(u.accepted for u in x.updates),
                          max_residual=max(u.residual_norm for u in x.updates)) for x in mps.history],
        nn_history=[dict(energy=x.energy, accepted_updates=sum(u.accepted for u in x.updates))
                    for x in result.history])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--integrals', type=Path)
    parser.add_argument('--cap', type=int, default=30)
    parser.add_argument('--sweeps', type=int, default=6)
    args = parser.parse_args()
    records = run() if args.integrals is None else run_molecular(args.integrals, args.cap, args.sweeps)
    if args.output:
        args.output.write_text(json.dumps(records, indent=2)+'\n')
    elif args.integrals is not None:
        print(json.dumps(records, indent=2))
