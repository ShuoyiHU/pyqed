#!/usr/bin/env python3
"""Same-start medium LETTA cases with larger inner iteration budgets."""
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys

CASES = tuple((model, shape, bond)
              for model in ('bose_hubbard', 'heisenberg')
              for shape in ((3, 3), (4, 3)) for bond in (2, 3))


def driver(root):
    path = root / 'source/pyqed/_letta_one_site_opt/benchmarks/2D/cluster_sept27/run_compression.py'
    spec = importlib.util.spec_from_file_location('medium_high_common', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CASES = CASES
    original_plan = module.make_plan

    def make_plan(args):
        two_sweeps = int(os.environ.get('TWO_SITE_SWEEPS', '20'))
        if two_sweeps < 1:
            raise ValueError('TWO_SITE_SWEEPS must be positive')
        result = original_plan(args)
        plan = json.loads(args.output.read_text())
        for task in plan['tasks']:
            if task['algorithm'] == 'two-site':
                task['max_sweeps'] = two_sweeps
        plan['protocol'] = dict(
            description='Larger inner budgets; frozen numerical source and identical starts. Recovery revision is recorded in the bundle manifest.',
            one_site_cbe_sweep_cap=args.sweeps, two_site_sweep_cap=two_sweeps,
            compression='ALS100/LSMR2000 by default',
            refinement_rounds_per_start=args.energy_refinement_iterations,
            early_stopping=True, seed=731)
        module.save(args.output, plan)
        return result

    module.make_plan = make_plan
    original_run = module.run_task
    original_summary = module.compression_summary

    def run_task(args):
        tasks = json.loads(args.plan.read_text())['tasks']
        if not 0 <= args.task_index < len(tasks):
            raise ValueError('Task index is outside the plan.')
        task = tasks[args.task_index]
        cap = task['energy_refinement_max_iterations']

        def summary(updates, algorithm):
            record = original_summary(updates, algorithm)
            prefix = 'cbe_' if algorithm == 'cbe' else ''
            counts = [getattr(u, prefix + 'energy_refinement_iterations', 0) for u in updates]
            starts = Counter(getattr(u, prefix + 'energy_refinement_start', None)
                             for u in updates)
            record['energy_refinement'] = dict(
                cap_per_start=cap, combined_rounds_sum=sum(counts),
                combined_rounds_max=max(counts, default=0),
                both_starts_hit_cap=sum(n >= 2 * cap for n in counts),
                selected_starts={str(k):v for k,v in starts.items() if k is not None},
                note='Combined iterations cover two starts; fewer than 2*cap does not prove either converged.')
            recoveries = [dict(site=u.site, reason=u.cbe_recovery_reason,
                               one_site_rejected=u.cbe_recovery_rejected)
                          for u in updates if getattr(u, 'cbe_recovery_reason', None)]
            record['numerical_recovery'] = dict(
                count=len(recoveries),
                rejected_one_site_steps=sum(r['one_site_rejected'] for r in recoveries),
                steps=recoveries)
            return record

        module.compression_summary = summary
        return original_run(args)

    module.run_task = run_task
    return module


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(__file__).resolve().parent
    if '--root' in argv:
        root = Path(argv[argv.index('--root')+1]).resolve()
    elif argv and argv[0] != 'collect':
        argv += ['--root', str(root)]
    if argv and argv[0] == 'plan':
        defaults = {'--seeds': ['731'], '--profiles': ['als100-lsmr2000'],
                    '--sweeps': ['2000'], '--energy-refinement-iterations': ['256']}
        for option, values in defaults.items():
            if option not in argv:
                argv += [option, *values]
    return driver(root).main(argv)


if __name__ == '__main__':
    sys.exit(main())
