"""Run actual small jobs through the frozen bundle and copied Slurm launcher."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
LAUNCH = REPO/'pyqed/_letta_one_site_opt/benchmarks/2D/cluster_sept27'


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp('frozen_compression')/'bundle'
    spec = importlib.util.spec_from_file_location('stage_compression', LAUNCH/'stage_bundle.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.stage(REPO, root)
    return root


def environment(root, **extra):
    return dict(os.environ, LETTA_PYTHON=sys.executable, BENCHMARK_ROOT=str(root),
                OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONWARNINGS='ignore',
                **extra)


def command(root, *args, **env):
    result = subprocess.run(['bash', str(root/'submit_compression.sh'), *args],
                            env=environment(root, **env), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    return result


def test_default_submission_has_correct_resources_and_full_matrix(bundle, tmp_path):
    fake = tmp_path/'sbatch'
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\nprintf "%s\\n" "$BENCHMARK_ROOT" > "$CAPTURE_ROOT"\necho 12345\n')
    fake.chmod(0o755)
    capture, root_capture = tmp_path/'args', tmp_path/'root'
    command(bundle, 'submit', PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],
            CAPTURE_ARGS=str(capture), CAPTURE_ROOT=str(root_capture), RUN_DIR=str(tmp_path/'run'))
    args = capture.read_text().splitlines()
    assert args[args.index('-p')+1]=='gubing'
    assert args[args.index('-q')+1]=='huge'
    assert '--time=0' in args and '--array=0-407' in args
    assert '--mem=64G' in args and '--cpus-per-task=1' in args
    assert not any('%' in x for x in args if x.startswith('--array='))
    assert root_capture.read_text().strip()==str(bundle)
    plan = json.loads((tmp_path/'run/plan.json').read_text())
    tasks = plan['tasks']
    assert len(tasks)==408
    assert len({t['case'] for t in tasks})==24
    assert all(t['max_sweeps']==100 for t in tasks)
    assert {t['profile'] for t in tasks}=={'none','als-default','als40-lsmr40',
        'als4-lsmr400','als40-lsmr400','als100-lsmr2000','variable-projection','joint-ls','grassmann-newton'}
    assert max(t['memory']['suggested_memory_gib'] for t in tasks)<64


def test_copied_slurm_workers_execute_all_profiles_and_collect(bundle, tmp_path):
    run = tmp_path/'run'
    command(bundle, 'plan', RUN_DIR=str(run), CASES='bose_hubbard:2x2:D2', SEEDS='731',
            SWEEPS='2', NONLINEAR_ITERATIONS='10', WORKSPACE_MB='128')
    tasks = json.loads((run/'plan.json').read_text())['tasks']
    assert len(tasks)==17
    spool = tmp_path/'slurm_spool'
    spool.mkdir()
    copied = spool/'slurm_script'
    shutil.copyfile(bundle/'submit_compression.sh', copied)
    for task in tasks:
        result = subprocess.run(['bash', str(copied), 'worker'], cwd=spool,
                                env=environment(bundle, RUN_DIR=str(run), SLURM_ARRAY_TASK_ID=str(task['index'])),
                                capture_output=True, text=True)
        assert result.returncode==0, result.stdout+'\n'+result.stderr
    reports = [json.loads(p.read_text()) for p in (run/'results').rglob('*.json')]
    assert len(reports)==17
    assert all(r['status']=='completed' and r['history'] for r in reports)
    assert len({r['initial_tensor_hash'] for r in reports})==1
    assert len({r['fingerprint'] for r in reports})==1
    for r in reports:
        profile = r['task']['profile']
        if profile in ('variable-projection','joint-ls','grassmann-newton'):
            assert any(s['compression']['used_solvers'].get(profile,0)>0 for s in r['history'])
    command(bundle, 'collect', str(run))
    summary = json.loads((run/'summary.json').read_text())
    assert len(summary)==17
    assert all(r['observed_initial_states_match'] and r['plan_matches'] for r in summary)
    assert all(r['energy_minus_completed_one_site'] is not None for r in summary)
    # Re-running an array index must not destroy a finished record.
    first = run/'results'/tasks[0]['case']/'one-site__none.json'
    before = first.read_bytes()
    rerun = subprocess.run(['bash',str(copied),'worker'],env=environment(bundle,RUN_DIR=str(run),SLURM_ARRAY_TASK_ID='0'),capture_output=True)
    assert rerun.returncode != 0 and first.read_bytes()==before


def test_preflight_rejects_changed_source(bundle):
    file = bundle/'source/pyqed/davidson.py'
    content = file.read_bytes()
    try:
        file.write_bytes(content+b'\n')
        result = subprocess.run(['bash',str(bundle/'submit_compression.sh'),'preflight'],
                                env=environment(bundle), capture_output=True,text=True)
        assert result.returncode != 0
        assert 'Frozen bundle verification failed' in result.stderr
    finally:
        file.write_bytes(content)
