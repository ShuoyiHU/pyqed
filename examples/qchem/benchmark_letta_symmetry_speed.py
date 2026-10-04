"""Matched-state / matched-energy symmetry benchmark for spatial orbitals.

The Abelian baseline is pyqed's masked-dense LETTA, not an optimized external
block-sparse DMRG. The untied reduced baseline is fixed-sector one-site MPS
optimization. Report these distinctions with every timing table.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from pyscf import fci

from pyqed._letta_one_site_opt import (
    LatticeLETTA, ReducedLatticeLETTA, ReducedFrontier, LETTADMROptions, letta_dmrg,
)
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.orbital_ordering import tie_neighborhoods
from pyqed.mps.nonabelian.coupling import clebsch_gordan, ordered_two_m_values


def expand_to_abelian(state, symmetry):
    """Exact benchmark/export conversion, never called by the SU(2) solver.

    Expand every *owned* physical spin with CG coefficients. Map repeated
    up/down conditioning indices to the same invariant single-occupancy label.
    The virtual multiplicity allocation is preserved, including all magnetic
    components of every internal multiplet. Select the requested boundary M.
    """
    if state.physical_basis.dense_dim != 4:
        raise ValueError('this benchmark converter requires spatial orbitals')
    target_n, target_m = symmetry.sector
    if target_n != state.symmetry.sector.charge or target_m not in range(-state.target_two_j, state.target_two_j+1, 2):
        raise ValueError('Abelian boundary does not belong to the target multiplet')
    def layout(sectors, final=False):
        entries = [(q, a, m) for q, r in Counter(sectors).items() for a in range(r)
                   for m in ordered_two_m_values(q.irrep) if not final or m == target_m]
        return entries, {entry: i for i, entry in enumerate(entries)}
    cuts = [layout(state.left_virtual_sectors(0))]
    cuts.extend(layout(state.right_virtual_sectors(i), i == state.nsites-1) for i in range(state.nsites))
    physical = [(state.physical_basis.sectors[0], 0),
                (state.physical_basis.sectors[1], 1),
                (state.physical_basis.sectors[1], -1),
                (state.physical_basis.sectors[2], 0)]
    label = (0, 1, 1, 2)
    tensors = []
    for i in range(state.nsites):
        left, li = cuts[i]; right, ri = cuts[i+1]
        degree = len(state.site_neighborhood(i))
        dtype = np.result_type(*[a.dtype for a in state.tensors[i].values()])
        tensor = np.zeros((len(left),)+(4,)*degree+(len(right),), dtype=dtype)
        for (ql, qp, qr), block in state.tensors[i].items():
            for (lq, a, ml), lindex in li.items():
                if lq != ql:
                    continue
                for (rq, b, mr), rindex in ri.items():
                    if rq != qr:
                        continue
                    for p, (pq, mp) in enumerate(physical):
                        if pq != qp:
                            continue
                        c = clebsch_gordan(ql.irrep, qp.irrep, qr.irrep, ml, mp, mr)
                        if c == 0:
                            continue
                        for dependency in np.ndindex(*((4,)*(degree-1))):
                            source = (a, 0)+tuple(label[x] for x in dependency)+(b,)
                            tensor[(lindex, p)+dependency+(rindex,)] = c*block[source]
        tensors.append(tensor)
    charges = tuple(tuple((q.charge, m) for q, _a, m in entries)
                    for entries, _lookup in cuts[1:-1])
    return LatticeLETTA(state.lattice_shape, 4, tensors, neighborhoods=state.neighborhoods,
                       symmetry=symmetry, bond_charges=charges)


def run(norb=4, multiplicity=2, max_passes=8, accuracy=1e-7, *, problem=None, model=None):
    if norb % 2:
        raise ValueError('the first timing suite uses even-electron singlets')
    if problem is None:
        h = -np.eye(norb, k=1)-np.eye(norb, k=-1)
        g = np.zeros((norb,)*4)
        for i in range(norb):
            g[i, i, i, i] = 4.
        problem = ElectronicProblem(h, g, (norb//2, norb//2))
    p = problem
    norb, h, g = p.norb, p.h1, p.eri
    if p.nelec[0] != p.nelec[1]:
        raise ValueError('this timing suite requires a singlet reference')
    exact, ci = fci.direct_spin1.kernel(h, g, norb, p.nelec, ecore=p.ecore, verbose=0)
    s2 = fci.spin_op.spin_square(ci, norb, p.nelec)[0]
    if abs(s2) > 1e-8:
        raise ValueError('FCI reference is not a singlet')
    sym = p.symmetry('su2')
    untied = ReducedLatticeLETTA.random((1, norb), symmetry=sym,
        neighborhoods=tuple((i,) for i in range(norb)), multiplets_per_sector=multiplicity, seed=71)
    tied = ReducedLatticeLETTA.from_mps(ReducedFrontier.from_state(untied).to_mps(untied),
        symmetry=sym, neighborhoods=tie_neighborhoods(norb, nearest=True))
    abelian = expand_to_abelian(tied, p.symmetry('n_sz'))
    # Small-system validation only, outside timed solver execution.
    embedding_error = float(np.linalg.norm(tied.state_vector()-abelian.state_vector()))
    if embedding_error > 1e-12:
        raise AssertionError('Abelian/SU(2) initial states differ')
    start = perf_counter(); native_h = p.su2_mpo(); native_h.native_mpo(sym.physical_basis)
    native_setup = perf_counter()-start
    # Dense molecular ERIs can exceed the CP-SVD builder's temporary storage
    # budget. Use the repository's compact graph AutoMPO for archived molecules.
    abelian_backend = 'svd' if model is None else 'symbolic'
    start = perf_counter(); abelian_h = p.mpo(backend=abelian_backend)
    abelian_setup = perf_counter()-start
    records = []
    # Time the actual tied SU(2) task before the untied control can warm its
    # recoupling/transfer caches. Each CLI run is a fresh Python process.
    for label, initial, mpo, setup in [('su2_nn', tied, native_h, native_setup),
                                     ('n_sz_nn', abelian, abelian_h, abelian_setup),
                                     ('su2_untied', untied, native_h, native_setup)]:
        state, history, elapsed = initial, [], 0.
        for sweep in range(max_passes):
            options = LETTADMROptions(max_sweeps=1, tolerance=1e-12, matrix_free=True,
                dense_solver_threshold=32, gauge_mode='frontier',
                start_direction='lr' if sweep % 2 == 0 else 'rl')
            start = perf_counter(); result = letta_dmrg(mpo, state=state, options=options)
            elapsed += perf_counter()-start
            state = result.state
            error = float(result.energy-exact)
            if error < -1e-8:
                raise AssertionError('energy below FCI')
            row = dict(pass_index=sweep+1, energy=float(result.energy), error_hartree=error,
                       cumulative_solver_seconds=elapsed)
            history.append(row)
            print(json.dumps(dict(method=label, **row)), flush=True)
            if error <= accuracy:
                break
        records.append(dict(method=label, reached_accuracy=history[-1]['error_hartree'] <= accuracy,
            setup_seconds=setup, solver_seconds=elapsed, total_seconds=setup+elapsed,
            history=history, final_bond_dimensions=state.bond_dimensions))
    return dict(model=model or 'open Hubbard chain t=1 U=4', norb=norb, nelec=p.nelec,
        seed=71, fci_energy=float(exact), accuracy_hartree=accuracy,
        initial_multiplet_dimensions=tied.bond_dimensions,
        initial_magnetic_dimensions=tied.magnetic_bond_dimensions,
        embedding_error=embedding_error, abelian_mpo_backend=abelian_backend, records=records,
        limitations=['Abelian implementation is masked dense, not optimized block sparse.',
                     'Untied SU(2) baseline is fixed-sector one-site MPS optimization.',
                     'SU(2) ties condition on occupancy multiplets; Abelian ties can resolve spin components.',
                     'Each pass reenters the public solver; timings include its initialization and final energy check.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--norb', type=int, default=4)
    parser.add_argument('--multiplicity', type=int, default=2)
    parser.add_argument('--max-passes', type=int, default=8)
    parser.add_argument('--accuracy', type=float, default=1e-7)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--integrals', type=Path, help='Archived ElectronicProblem NPZ instead of Hubbard integrals')
    args = parser.parse_args()
    problem = None
    if args.integrals:
        data = np.load(args.integrals)
        problem = ElectronicProblem(data['h1'], data['eri'], tuple(data['nelec']),
                                    float(data['ecore']), tuple(data['orbital_order']))
    report = run(args.norb, args.multiplicity, args.max_passes, args.accuracy,
                 problem=problem, model=None if args.integrals is None else args.integrals.stem)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
