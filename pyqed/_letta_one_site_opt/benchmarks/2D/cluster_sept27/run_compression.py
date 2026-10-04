#!/usr/bin/env python3
"""Frozen-source, same-start LETTA compression comparisons on independent jobs."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import platform
import resource
import signal
import sys
from time import perf_counter
import traceback
from unittest.mock import patch

PROFILES = {
    "als-default": {"solver": "als"},
    "als40-lsmr40": {"solver": "als", "als_max_iterations": 40, "lsmr_max_iterations": 40},
    "als4-lsmr400": {"solver": "als", "als_max_iterations": 4, "lsmr_max_iterations": 400},
    "als40-lsmr400": {"solver": "als", "als_max_iterations": 40, "lsmr_max_iterations": 400},
    "als100-lsmr2000": {"solver": "als", "als_max_iterations": 100, "lsmr_max_iterations": 2000},
    "variable-projection": {"solver": "variable-projection", "als_max_iterations": 40, "lsmr_max_iterations": 400},
    "joint-ls": {"solver": "joint-ls", "als_max_iterations": 40, "lsmr_max_iterations": 400},
    "grassmann-newton": {"solver": "grassmann-newton", "als_max_iterations": 40, "lsmr_max_iterations": 400},
}
CASES = (
    ("bose_hubbard", (2, 2), 2),
    ("bose_hubbard", (3, 2), 3),
    ("bose_hubbard", (3, 2), 4),
    ("bose_hubbard", (3, 3), 4),
    ("bose_hubbard", (3, 3), 5),
    ("bose_hubbard", (4, 3), 4),
    ("ising", (3, 3), 4),
    ("ising", (4, 3), 4),
    ("heisenberg", (3, 3), 4),
    ("heisenberg", (4, 3), 4),
    ("fermi_hubbard", (3, 2), 3),
    ("fermi_hubbard", (3, 3), 3),
)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def previous_driver(source):
    path = Path(source) / 'pyqed/_letta_one_site_opt/benchmarks/2D/cluster_sept18/run_large_2d.py'
    spec = importlib.util.spec_from_file_location('letta_resource_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_bundle(root):
    root = Path(root).resolve()
    path = root / 'BUNDLE_MANIFEST.json'
    manifest = json.loads(path.read_text())
    failures = []
    for entry in manifest['files']:
        file = root / entry['path']
        if not file.is_file() or digest(file) != entry['sha256']:
            failures.append(entry['path'])
    if failures:
        raise RuntimeError(f'Frozen bundle verification failed: {failures}')
    return dict(bundle_sha256=digest(path), verified_files=len(manifest['files']))


def runtime(root):
    verified = verify_bundle(root)
    source = Path(root).resolve() / 'source'
    class FrozenPyqedFinder:
        # An editable-install finder can otherwise supply missing optional
        # submodules after the ordinary path finder has exhausted the bundle.
        def find_spec(self, fullname, path=None, target=None):
            if fullname != 'pyqed' and not fullname.startswith('pyqed.'):
                return None
            spec = PathFinder.find_spec(fullname, [str(source)] if fullname == 'pyqed' else path)
            if spec is None:
                raise ModuleNotFoundError(f'{fullname} is not in the frozen bundle', name=fullname)
            if spec.origin and spec.origin not in ('built-in', 'frozen'):
                Path(spec.origin).resolve().relative_to(source)
            return spec
    sys.meta_path.insert(0, FrozenPyqedFinder())
    environment = previous_driver(source).load_runtime(source)
    from pyqed._letta_one_site_opt import MetricCompressionOptions, LETTADMROptions
    from pyqed._letta_two_site_opt import LETTATwoSiteOptions
    for profile in PROFILES.values():
        c = MetricCompressionOptions(**profile)
        LETTADMROptions(compression=c)
        LETTATwoSiteOptions(compression=c)
    import pyqed._letta_compression as module
    Path(module.__file__).resolve().relative_to(source)
    return dict(environment, **verified, compression_source=module.__file__)


def make_plan(args):
    root = args.root.resolve()
    verified = verify_bundle(root)
    memory = previous_driver(root / 'source').memory_estimate
    tasks = []
    for model, shape, bond in CASES:
        if args.models and model not in args.models:
            continue
        if args.cases and f'{model}:{shape[0]}x{shape[1]}:D{bond}' not in args.cases:
            continue
        for seed in dict.fromkeys(args.seeds):
            case = f'{model}_{shape[0]}x{shape[1]}_D{bond}_seed{seed}'
            for algorithm in args.algorithms:
                for profile in (['none'] if algorithm == 'one-site' else args.profiles):
                    config = {} if profile == 'none' else dict(PROFILES[profile])
                    if profile != 'none':
                        config.update(max_iterations=args.nonlinear_iterations,
                                      tolerance=1e-10, max_workspace_mb=args.workspace_mb)
                    estimate = memory(model, shape, bond)
                    estimate['suggested_memory_gib'] += args.workspace_mb / 1024
                    tasks.append(dict(index=len(tasks), case=case, model=model,
                        shape=list(shape), bond_dim=bond, seed=seed, algorithm=algorithm,
                        profile=profile, compression=config, max_sweeps=args.sweeps,
                        energy_refinement_max_iterations=getattr(args, 'energy_refinement_iterations', 32),
                        tolerance=1e-12, metric_tolerance=1e-10, memory=estimate))
    if not tasks:
        raise ValueError('No cases selected.')
    initial_catalog = root / 'INITIAL_STATES.json'
    if initial_catalog.exists():
        entries = json.loads(initial_catalog.read_text())
        for task in tasks:
            if task['case'] not in entries:
                raise ValueError(f"No frozen initial state for {task['case']}")
            task['initial_state'] = entries[task['case']]
    if min(args.sweeps, args.nonlinear_iterations, args.workspace_mb, args.memory_gib,
           getattr(args, 'energy_refinement_iterations', 32)) <= 0:
        raise ValueError('Budgets must be positive.')
    blocked = [t['case'] for t in tasks if t['memory']['suggested_memory_gib'] > args.memory_gib]
    if blocked:
        raise ValueError(f'Estimated memory exceeds requested memory for {sorted(set(blocked))}')
    if args.output.exists():
        raise FileExistsError(args.output)
    save(args.output, dict(schema=1, created=datetime.now(timezone.utc).isoformat(),
        bundle_sha256=verified['bundle_sha256'], memory_gib=args.memory_gib,
        cpus=args.cpus, tasks=tasks))
    print(f'{len(tasks)} tasks; {len({t["case"] for t in tasks})} case/seed combinations; array 0-{len(tasks)-1}', flush=True)
    return 0


def initial_state(root, model, task):
    """Use serialized tensors when a plan specifies a shared initial state."""
    from pyqed._letta_one_site_opt.benchmarks.condensed_runner import (
        make_shared_initial_state, _hash_arrays)
    entry = task.get('initial_state')
    if entry is None:
        return make_shared_initial_state(model, bond_dim=task['bond_dim'], seed=task['seed'])
    import numpy as np
    from types import SimpleNamespace
    from pyqed._letta_one_site_opt import LatticeLETTA
    path = root / entry['path']
    if digest(path) != entry['sha256']:
        raise ValueError('Frozen initial-state file hash mismatch')
    with np.load(path, allow_pickle=False) as data:
        tensors = [data[f'tensor_{i}'].copy() for i in range(model.nsites)]
    if _hash_arrays(tensors) != entry['tensor_hash']:
        raise ValueError('Frozen initial tensor hash mismatch')
    state = LatticeLETTA(model.lattice_shape, model.physical_dim, tensors)
    # Construction validates and normalizes. Restore the already normalized
    # serialized arrays so host-dependent rounding cannot change input hashes.
    state.tensors = tensors
    if abs(float(state.norm())-1.) > 1e-10:
        raise ValueError('Frozen initial state is not normalized')
    return SimpleNamespace(letta=state, fingerprint=entry['fingerprint'],
                           energy=float(state.expectation(model.mpo)))


def compression_summary(updates, algorithm):
    fits = ([u.compression_diagnostics for u in updates if u.compression_diagnostics]
            if algorithm == 'two-site' else
            [d for u in updates for d in u.cbe_compression_diagnostics])
    used = Counter(d['used_solver'] for d in fits)
    reasons = Counter(d['fallback_reason'] for d in fits if d.get('fallback_reason'))
    finite_loss = lambda key: [float(d[key]) for d in fits if d.get(key) is not None]
    return dict(fits=len(fits), used_solvers=dict(used), fallback_reasons=dict(reasons),
        fallback_count=sum(reasons.values()),
        optimizer_success=sum(d.get('optimizer_success') is True for d in fits),
        optimizer_not_success=sum(d.get('optimizer_success') is False for d in fits),
        evaluations=sum(d.get('evaluations', 0) for d in fits),
        rejected_candidates=sum(d.get('rejected_candidates', 0) for d in fits),
        initial_loss_sum=sum(finite_loss('initial_loss')), final_loss_sum=sum(finite_loss('final_loss')),
        max_workspace_estimate_mb=max((d.get('estimated_workspace_bytes', 0)/1024**2 for d in fits), default=0))


def run_task(args):
    root, run_dir = args.root.resolve(), args.plan.resolve().parent
    plan = json.loads(args.plan.read_text())
    if not 0 <= args.task_index < len(plan['tasks']):
        raise ValueError('Task index is outside the plan.')
    task = plan['tasks'][args.task_index]
    if task['index'] != args.task_index:
        raise ValueError('Task index mismatch')
    folder = run_dir / 'results' / task['case']
    label = task['algorithm'] + '__' + task['profile']
    output = folder / (label + '.json')
    folder.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents an accidental rerun from overwriting progress.
    with output.open('x') as file:
        file.write('{}\n')
    report = dict(task=task, status='starting', history=[], started=datetime.now(timezone.utc).isoformat(),
                  host=platform.node(), job_id=os.getenv('SLURM_JOB_ID'),
                  array_job_id=os.getenv('SLURM_ARRAY_JOB_ID'), array_task_id=os.getenv('SLURM_ARRAY_TASK_ID'),
                  plan_sha256=digest(args.plan))
    start = perf_counter()
    observer_seconds = 0.
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    save(output, report)
    try:
        report['runtime'] = runtime(root)
        if report['runtime']['bundle_sha256'] != plan['bundle_sha256']:
            raise RuntimeError('Plan and current frozen bundle differ.')
        from pyqed._letta_one_site_opt import LETTADMROptions, MetricCompressionOptions, letta_dmrg
        from pyqed._letta_one_site_opt import solver as one_solver
        from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg
        from pyqed._letta_two_site_opt import solver as two_solver
        from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
        from pyqed._letta_one_site_opt.benchmarks.condensed_runner import _hash_arrays
        import numpy as np
        model = build_model(task['model'], '2d', tuple(task['shape']))
        initial = initial_state(args.root, model, task)
        original_hash = _hash_arrays(initial.letta.tensors)
        common = dict(max_sweeps=task['max_sweeps'], tolerance=task['tolerance'],
                      metric_tolerance=task['metric_tolerance'], eigensolver_tolerance=1e-10, verbosity=1)
        compression = MetricCompressionOptions(**task['compression'])
        if task['algorithm'] == 'two-site':
            options = LETTATwoSiteOptions(**common, split_method='metric-energy', compression=compression,
                energy_refinement_max_iterations=task.get('energy_refinement_max_iterations', 32))
            module, name = two_solver, '_pair_sweep'
        else:
            options = LETTADMROptions(**common, compression=compression,
                                     cbe_enabled=task['algorithm'] == 'cbe', cbe_selector='shrewd',
                                     cbe_energy_refinement_max_iterations=task.get('energy_refinement_max_iterations', 32),
                                     cbe_baseline_guard_fraction=0., cbe_expansion_dimension=1)
            module, name = one_solver, '_cached_mpo_sweep'
        report.update(status='running', parameters=dict(model.parameters), options=asdict(options),
                      fingerprint=initial.fingerprint, initial_tensor_hash=original_hash,
                      initial_energy=initial.energy)
        save(output, report)
        solve_start = perf_counter()
        original = getattr(module, name)
        def observed(state, *positional, **keywords):
            nonlocal observer_seconds
            result = original(state, *positional, **keywords)
            solver_seconds = perf_counter()-solve_start-observer_seconds
            observe_start = perf_counter()
            updates, cached_energy = result[:2]
            energy = float(state.expectation(model.mpo))
            previous = report['history'][-1]['energy'] if report['history'] else initial.energy
            row = dict(sweep=len(report['history'])+1, energy=float(energy),
                       cached_energy=float(cached_energy),
                       fresh_energy_check_passed=bool(np.isfinite(energy)
                           and energy <= previous + options.energy_increase_tolerance),
                       solver_seconds=solver_seconds, energy_change=float(energy)-previous,
                       hamiltonian_applications=sum(u.hamiltonian_applications for u in updates),
                       compression=compression_summary(updates, task['algorithm']))
            row['cbe_baseline_selected'] = sum(getattr(u, 'cbe_baseline_selected', False) for u in updates)
            row['phases'] = {}
            for update in updates:
                for key, value in (getattr(update, 'cbe_timings', None) or {}).items():
                    row['phases'][key] = row['phases'].get(key, 0.)+value
            report['history'].append(row)
            report['updated'] = datetime.now(timezone.utc).isoformat()
            save(output, report)
            print(json.dumps(row, allow_nan=False), flush=True)
            observer_seconds += perf_counter()-observe_start
            return result
        with patch.object(module, name, observed):
            if task['algorithm'] == 'two-site':
                result = letta_two_site_dmrg(model.mpo, state=initial.letta,
                                             bond_dim=task['bond_dim'], options=options)
            else:
                result = letta_dmrg(model.mpo, state=initial.letta, options=options)
        seconds = perf_counter()-solve_start-observer_seconds
        physical_energy = float(result.state.expectation(model.mpo))
        if abs(physical_energy-result.energy) > 1e-8*max(1., abs(physical_energy)):
            raise AssertionError('Final contracted energy disagrees with sweep energy.')
        if _hash_arrays(initial.letta.tensors) != original_hash:
            raise AssertionError('Solver mutated its shared initial state.')
        np.savez_compressed(folder / (label+'__final.npz'),
                            **{f'tensor_{i}': a for i, a in enumerate(result.state.tensors)})
        report.update(status='completed', energy=result.energy, physical_energy=physical_energy,
                      message=result.message,
                      solver_seconds=seconds, observer_seconds=observer_seconds,
                      sweeps=result.sweeps, converged=result.converged,
                      max_sweep_energy_increase=max((r['energy_change'] for r in report['history']), default=0.))
        return 0
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        report['wall_seconds'] = perf_counter()-start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['max_rss_gib'] = rss/(1024**3 if sys.platform == 'darwin' else 1024**2)
        save(output, report)
        print(f"RESULT {report['status']} {output}", flush=True)


def collect(args):
    plan = json.loads(args.plan.read_text())
    plan_hash = digest(args.plan)
    rows, hashes = [], {}
    for task in plan['tasks']:
        file = args.plan.parent/'results'/task['case']/(task['algorithm']+'__'+task['profile']+'.json')
        r = json.loads(file.read_text()) if file.exists() else {}
        history = r.get('history', [])
        last = history[-1] if history else {}
        used, reasons = Counter(), Counter()
        for sweep in history:
            used.update(sweep['compression']['used_solvers'])
            reasons.update(sweep['compression']['fallback_reasons'])
        if r.get('fingerprint'):
            hashes.setdefault(task['case'], set()).add((r['fingerprint'], r.get('initial_tensor_hash')))
        rows.append(dict(case=task['case'], algorithm=task['algorithm'], profile=task['profile'],
            status=r.get('status', 'not_started'), sweeps=r.get('sweeps', len(history)),
            energy=r.get('energy', last.get('energy')), solver_seconds=r.get('solver_seconds',last.get('solver_seconds')),
            converged=r.get('converged'), max_rss_gib=r.get('max_rss_gib'),
            compression_fits=sum(used.values()), nonlinear_fits=sum(v for k,v in used.items() if k in ('variable-projection','joint-ls','grassmann-newton')),
            als_fallbacks=sum(reasons.values()), used_solvers=json.dumps(dict(used), sort_keys=True),
            fallback_reasons=json.dumps(dict(reasons), sort_keys=True),
            plan_matches=r.get('plan_sha256') == plan_hash if r else None, result=str(file)))
    baselines = {r['case']: r['energy'] for r in rows if r['algorithm']=='one-site' and r['status']=='completed'}
    for row in rows:
        base = baselines.get(row['case'])
        row['energy_minus_completed_one_site'] = row['energy']-base if base is not None and row['energy'] is not None else None
        row['observed_initial_states_match'] = len(hashes.get(row['case'], ()))==1 if hashes.get(row['case']) else None
    path = args.plan.parent/'summary.csv'
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save(args.plan.parent/'summary.json', rows)
    print(json.dumps(dict(Counter(r['status'] for r in rows)), indent=2))
    print(path)
    print('running means no final record yet; check Slurm for killed jobs. Partial energies may be from different sweep counts.')
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    default_root = Path(__file__).resolve().parent
    q = sub.add_parser('preflight')
    q.add_argument('--root', type=Path, default=default_root)
    q = sub.add_parser('plan')
    q.add_argument('--root', type=Path, default=default_root)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--seeds', nargs='+', type=int, default=[731, 1735])
    q.add_argument('--models', nargs='+', choices=sorted({c[0] for c in CASES}))
    q.add_argument('--cases', nargs='+')
    q.add_argument('--algorithms', nargs='+', choices=['one-site','cbe','two-site'], default=['one-site','cbe','two-site'])
    q.add_argument('--profiles', nargs='+', choices=list(PROFILES), default=list(PROFILES))
    q.add_argument('--sweeps', type=int, default=100)
    q.add_argument('--nonlinear-iterations', type=int, default=100)
    q.add_argument('--energy-refinement-iterations', type=int, default=32)
    q.add_argument('--workspace-mb', type=float, default=1024.)
    q.add_argument('--memory-gib', type=float, default=64.)
    q.add_argument('--cpus', type=int, default=1)
    q = sub.add_parser('run')
    q.add_argument('--root', type=Path, default=default_root)
    q.add_argument('--plan', type=Path, required=True)
    q.add_argument('--task-index', type=int, required=True)
    q = sub.add_parser('collect')
    q.add_argument('--plan', type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == 'preflight':
        print(json.dumps(runtime(args.root), indent=2))
        return 0
    return {'plan':make_plan, 'run':run_task, 'collect':collect}[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
