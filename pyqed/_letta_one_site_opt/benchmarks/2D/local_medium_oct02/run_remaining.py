"""Run unfinished frozen tasks with durable logs and a coordinator heartbeat.

Launch this coordinator in its own OS session. Existing completed records are
preserved; interrupted reports and logs are archived before restarting a task.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    plan = json.loads(args.plan.read_text())
    folder = args.plan.parent
    root = folder.parent
    old = json.loads((folder / 'batch.json').read_text()) if (folder / 'batch.json').exists() else {}
    archive = folder / 'interrupted_attempts' / datetime.now().strftime('%Y%m%d-%H%M%S')
    archive.mkdir(parents=True)
    save(archive / 'previous_batch.json', old)
    tasks = []
    preserved = []
    for task in plan['tasks']:
        result = folder / 'results' / task['case'] / f"{task['algorithm']}__{task['profile']}.json"
        report = json.loads(result.read_text()) if result.exists() else {}
        if report.get('status') == 'completed':
            preserved.append(task['index'])
            continue
        tasks.append(task)
        if result.exists():
            result.rename(archive / f"{task['index']:03}_{result.name}")
        log = folder / 'logs' / f"{task['index']:03}.log"
        if log.exists():
            log.rename(archive / log.name)
    # Finish the small Heisenberg comparisons before the larger Bose jobs.
    tasks.sort(key=lambda t: (t['model'] != 'heisenberg', t['bond_dim'], t['shape'], t['algorithm']))
    pending = deque(tasks)
    metadata = dict(status='running', pid=os.getpid(), started=now(),
                    preserved_completed=preserved, restarted_indices=[t['index'] for t in tasks],
                    archive=str(archive), completed=[], active=[], queued=[])
    active = {}
    env = dict(os.environ, PYTHONPATH=str(root / 'source'), PYTHONDONTWRITEBYTECODE='1',
               PYTHONUNBUFFERED='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MKL_NUM_THREADS='1')
    driver = root / 'source/pyqed/_letta_one_site_opt/benchmarks/2D/cluster_sept27/run_compression.py'
    spec = importlib.util.spec_from_file_location('medium_collect', Path(__file__).with_name('collect.py'))
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    last_collection = 0
    while active or pending:
        for i, (process, log, started) in list(active.items()):
            code = process.poll()
            if code is not None:
                log.close()
                metadata['completed'].append(dict(index=i, code=code, seconds=time.monotonic()-started))
                del active[i]
                print('FINISHED', i, code, now(), flush=True)
        while pending and len(active) < args.workers:
            task = pending.popleft()
            index = task['index']
            log = (folder / 'logs' / f'{index:03}.log').open('w')
            process = subprocess.Popen([sys.executable, str(driver), 'run', '--root', str(root),
                '--plan', str(args.plan), '--task-index', str(index)], env=env, cwd=root / 'source',
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            active[index] = (process, log, time.monotonic())
            print('START', index, process.pid, task['case'], task['algorithm'], now(), flush=True)
        metadata.update(heartbeat=now(), active=[dict(index=i,pid=p.pid) for i,(p,_,_) in active.items()],
                        queued=[t['index'] for t in pending])
        if not active and not pending:
            metadata['status'] = 'completed'
        save(folder / 'batch.json', metadata)
        if time.monotonic()-last_collection >= 60 or metadata['status'] == 'completed':
            try:
                collector.collect(args.plan, args.output)
            except Exception as error:
                print('COLLECTION ERROR', repr(error), flush=True)
            last_collection = time.monotonic()
        if active or pending:
            time.sleep(10)


if __name__ == '__main__':
    main()
