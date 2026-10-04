"""H2O active-space / multiplet-cap comparison of SU(2) MPS and NN LETTA.

FCI and full vectors are independent post-run diagnostics only. All variational
optimization uses native reduced MPO contractions. Run with single-thread BLAS.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
from pyscf import ao2mo, fci, gto, mcscf, scf

from pyqed._letta_one_site_opt import (
    ReducedLatticeLETTA, ReducedFrontier, LETTADMROptions, letta_dmrg,
)
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.orbital_ordering import tie_neighborhoods
from pyqed._letta_one_site_opt.benchmarks.qchem_active_space import (
    determinant_map, physical_diagnostics, apply_cas_vector, diagnostic_state_vector,
)
from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg


ATOM = 'O 0 0 0; H 0 0 0.96; H 0.9294217 0 -0.24038'


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def prepare(root, spaces):
    root.mkdir(parents=True, exist_ok=True)
    mf = None
    for electrons, orbitals in spaces:
        folder = root/f'cas_{electrons}_{orbitals}'
        folder.mkdir(exist_ok=True)
        path = folder/'integrals.npz'
        if path.exists():
            continue
        if mf is None:
            mol = gto.M(atom=ATOM, basis='6-31g', unit='Angstrom', spin=0, verbose=0)
            mf = scf.RHF(mol)
            mf.conv_tol = 1e-12
            mf.kernel()
            if not mf.converged:
                raise RuntimeError('RHF did not converge')
        cas = mcscf.CASCI(mf, orbitals, electrons)
        h, core = cas.get_h1eff()
        eri = ao2mo.restore(1, cas.get_h2eff(), orbitals)
        np.savez_compressed(path, h1=h, eri=eri, ecore=core, nelec=cas.nelecas,
                            orbital_order=np.arange(orbitals), mo_coeff=mf.mo_coeff)
        write_json(folder/'model.json', dict(atom=ATOM, unit='Angstrom', basis='6-31g',
            active_electrons=electrons, active_orbitals=orbitals, frozen_orbitals=cas.ncore,
            active_mo_indices=list(range(cas.ncore, cas.ncore+orbitals)),
            orbital_basis='RHF canonical, ascending energy, contiguous active window',
            orbital_optimization=False, rhf_energy=float(mf.e_tot), ecore=float(core),
            integral_sha256=hashlib.sha256(h.tobytes()+eri.tobytes()).hexdigest()))


def reference(problem):
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-14
    solver.conv_tol_residual = 1e-10
    solver.lindep = 1e-20
    solver.max_space = 100
    solver.max_cycle = 300
    energy, ci = solver.kernel(problem.h1, problem.eri, problem.norb, problem.nelec,
                               ecore=problem.ecore)
    if not solver.converged:
        raise RuntimeError('FCI did not converge')
    flats, signs = determinant_map(problem)
    vector = np.zeros(4**problem.norb)
    vector[flats] = ci*signs
    s2 = fci.spin_op.spin_square(ci, problem.norb, problem.nelec)[0]
    if abs(s2) > 1e-8:
        raise AssertionError('reference is not a singlet')
    return float(energy), vector


def solve(state, h, cap, method, max_passes, tolerance, folder):
    start = perf_counter()
    history, stable, cycle_stable = [], 0, 0
    previous = None
    for sweep in range(max_passes):
        direction = 'lr' if sweep % 2 == 0 else 'rl'
        if method == 'dmrg':
            # A Schmidt truncation need not lower E; always enforce the cap.
            # The following one-site phase restores monotone variational optimization.
            result = letta_two_site_dmrg(h, state=state, bond_dim=cap,
                options=LETTATwoSiteOptions(max_sweeps=1, tolerance=tolerance,
                    eigensolver_tolerance=1e-11, metric_tolerance=1e-12,
                    split_method='conditional-svd', reduced_sector_growth=True,
                    energy_increase_tolerance=float('inf'),
                    gauge_mode='frontier', truncation_max_iterations=12,
                    dense_solver_threshold=32, start_direction=direction))
        else:
            result = letta_dmrg(h, state=state, options=LETTADMROptions(
                max_sweeps=1, tolerance=tolerance, eigensolver_tolerance=1e-11,
                energy_increase_tolerance=1e-11, gauge_mode='frontier',
                dense_solver_threshold=32, start_direction=direction))
        state = result.state
        change = None if previous is None else abs(result.energy-previous)
        stable = stable+1 if change is not None and change <= tolerance else 0
        cycle_change = abs(result.energy-history[-2]['energy']) if len(history) >= 2 else None
        cycle_stable = cycle_stable+1 if method == 'dmrg' and cycle_change is not None and cycle_change <= tolerance else 0
        updates = result.history[-1].updates
        row = dict(pass_index=sweep+1, energy=result.energy, change=change, cycle_change=cycle_change,
            max_local_residual=max(u.residual_norm for u in updates),
            accepted=sum(u.accepted for u in updates), updates=len(updates),
            seconds=perf_counter()-start)
        history.append(row)
        print(json.dumps(dict(case=folder.name, cap=cap, method=method, **row)), flush=True)
        write_json(folder/f'progress_{method}_{cap}.json', history)
        previous = result.energy
        if stable >= 2 or cycle_stable >= 2:
            break
    return state, dict(energy=result.energy, seconds=perf_counter()-start,
        passes=len(history), energy_stationary=stable >= 2, sweep_cycle_stationary=cycle_stable >= 2, history=history)


def diagnostics(state, problem, exact, vector):
    row = physical_diagnostics(state, problem, exact, vector)
    row.update(multiplet_dimensions=state.bond_dimensions,
        magnetic_dimensions=state.magnetic_bond_dimensions,
        max_multiplets=max(state.bond_dimensions),
        max_magnetic=max(state.magnetic_bond_dimensions),
        reduced_parameter_count=sum(a.size for core in state.tensors for a in core.values()),
        sector_multiplicities=[[(q.charge, q.irrep.two_j, r) for q, r in Counter(b).items()]
                               for b in state.bond_sectors])
    if (not all(np.isfinite(row[k]) for k in ['energy_error', 'spin_squared', 'residual_norm'])
            or row['energy_error'] < -1e-8 or abs(row['spin_squared']) > 1e-8
            or row['sector_leakage'] > 1e-10):
        raise AssertionError(f'physical validation failed: {row}')
    return row


def run(folder, caps, max_passes, tolerance, seed):
    data = np.load(folder/'integrals.npz')
    p = ElectronicProblem(data['h1'], data['eri'], tuple(data['nelec']),
                          float(data['ecore']), tuple(data['orbital_order']))
    exact, vector = reference(p)
    residual = float(np.linalg.norm(apply_cas_vector(p, vector)-exact*vector))
    if residual > 2e-10:
        raise AssertionError(f'FCI residual too large: {residual}')
    write_json(folder/'reference.json', dict(energy=exact, residual_norm=residual))
    start = perf_counter()
    h = p.su2_mpo()
    h.native_mpo(p.symmetry('su2').physical_basis)
    setup = perf_counter()-start
    print(f'PREPARED {folder.name} FCI={exact:.14f} setup={setup:.3f}s', flush=True)
    records = []
    for cap in caps:
        record_path, checkpoint = folder/f'D{cap}.json', folder/f'D{cap}_dmrg.pkl'
        if record_path.exists() and checkpoint.exists():
            saved = json.loads(record_path.read_text())
            if saved['seed'] != seed:
                raise ValueError('use a separate output directory for each seed')
            if saved.get('reference_version', 1) < 2:
                for method in ('dmrg', 'letta'):
                    with (folder/f'D{cap}_{method}.pkl').open('rb') as stream:
                        state = pickle.load(stream)
                    saved[method].update(diagnostics(state, p, exact, vector))
                saved['fci_energy'] = exact
                saved['reference_version'] = 2
                write_json(record_path, saved)
            records.append(saved)
            if saved['dmrg']['energy_error'] < 1e-10 and saved['letta']['energy_error'] < 1e-10:
                break
            continue
        initial = ReducedLatticeLETTA.random((1, p.norb), symmetry=p.symmetry('su2'),
            neighborhoods=tuple((i,) for i in range(p.norb)),
            multiplets_per_sector=1, seed=seed)
        mps, mps_run = solve(initial, h, cap, 'dmrg', max_passes, tolerance, folder)
        # Optimize remaining fixed-sector coordinates after the adaptive split.
        mps, polish = solve(mps, h, cap, 'mps_polish', min(max_passes, 12), tolerance, folder)
        md = diagnostics(mps, p, exact, vector)
        if abs(md['total_energy']-polish['energy']) > 1e-8 or md['max_multiplets'] > cap:
            raise AssertionError(f'DMRG energy or cap validation failed: {md}, native={polish["energy"]}')
        tied = ReducedLatticeLETTA.from_mps(ReducedFrontier.from_state(mps).to_mps(mps),
            symmetry=p.symmetry('su2'), neighborhoods=tie_neighborhoods(p.norb, nearest=True))
        distance = float(np.linalg.norm(diagnostic_state_vector(tied)-diagnostic_state_vector(mps)))
        nn, nn_run = solve(tied, h, cap, 'letta_nn', max_passes, tolerance, folder)
        nd = diagnostics(nn, p, exact, vector)
        if (distance > 1e-11 or nd['total_energy'] > md['total_energy']+1e-9
                or abs(nd['total_energy']-nn_run['energy']) > 1e-8
                or nn.bond_sectors != mps.bond_sectors):
            raise AssertionError('LETTA embedding, energy, or allocation validation failed')
        row = dict(cap=cap, dimension_unit='SU2 multiplets', fci_energy=exact, reference_version=2,
            hamiltonian_setup_seconds=setup, embedding_distance=distance,
            initialization='random, one copy per reachable sector', seed=seed,
            dmrg=dict(**md, optimization=mps_run, polish=polish),
            letta=dict(**nd, optimization=nn_run))
        write_json(record_path, row)
        with checkpoint.open('wb') as stream:
            pickle.dump(mps, stream)
        with (folder/f'D{cap}_letta.pkl').open('wb') as stream:
            pickle.dump(nn, stream)
        records.append(row)
        print(f'DONE {folder.name} D={cap} MPS error={md["energy_error"]:.4e} '
              f'LETTA error={nd["energy_error"]:.4e}', flush=True)
        if md['energy_error'] < 1e-10 and nd['energy_error'] < 1e-10:
            break
    records = sorted((json.loads(path.read_text()) for path in folder.glob('D*.json')), key=lambda row: row['cap'])
    write_json(folder/'results.json', records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cas', nargs='+', default=['4,4', '6,6', '8,6', '8,8'])
    parser.add_argument('--caps', nargs='+', type=int, default=[4, 8, 12, 16, 24, 32, 48, 64])
    parser.add_argument('--max-passes', type=int, default=20)
    parser.add_argument('--tolerance', type=float, default=1e-10)
    parser.add_argument('--seed', type=int, default=71)
    parser.add_argument('--output', type=Path, default=Path('examples/qchem/water_su2_benchmark'))
    args = parser.parse_args()
    spaces = [tuple(map(int, x.split(','))) for x in args.cas]
    if args.caps != sorted(set(args.caps)) or min(args.caps) < 1:
        raise ValueError('caps must be positive, distinct, and increasing')
    prepare(args.output, spaces)
    for ne, no in spaces:
        run(args.output/f'cas_{ne}_{no}', args.caps, args.max_passes, args.tolerance, args.seed)
