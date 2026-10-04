"""Check rectangle plans and real workers loading frozen shared tensors."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

REPO=Path(__file__).resolve().parents[1]
LAUNCH=REPO/'pyqed/_letta_one_site_opt/benchmarks/2D/cluster_oct02'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    root=tmp_path_factory.mktemp('rectangles')/'bundle'
    cases=load(LAUNCH/'run_rectangles.py','rectangle_cases_test').CASES
    load(LAUNCH/'stage_bundle.py','rectangle_stage_test').stage(
        REPO,root,cases=cases+(('bose_hubbard',(2,2),2),))
    return root


def env(root,**extra):
    return dict(os.environ,LETTA_PYTHON=sys.executable,BENCHMARK_ROOT=str(root),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',**extra)


def test_rectangle_submission_resources_and_plan(bundle,tmp_path):
    fake=tmp_path/'sbatch'
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\necho 12345\n')
    fake.chmod(0o755)
    capture=tmp_path/'args'
    p=subprocess.run(['bash',str(bundle/'submit_rectangles.sh'),'submit'],
        env=env(bundle,PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],
                CAPTURE_ARGS=str(capture),RUN_DIR=str(tmp_path/'run')),
        capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    args=capture.read_text().splitlines()
    assert args[args.index('-p')+1]=='gubing' and args[args.index('-q')+1]=='huge'
    assert '--time=0' in args and '--array=0-79' in args and '--mem=256G' in args
    assert not any('%' in a for a in args if a.startswith('--array='))
    plan=json.loads((tmp_path/'run/plan.json').read_text())
    tasks=plan['tasks']
    assert len(tasks)==80 and len({t['case'] for t in tasks})==16
    assert {t['algorithm'] for t in tasks}=={'one-site','cbe','two-site'}
    assert {tuple(t['shape']) for t in tasks}=={(12,3),(12,4)}
    assert {t['profile'] for t in tasks}=={'none','als4-lsmr400','als40-lsmr400'}
    assert all(t['max_sweeps']==500 and t['memory']['suggested_memory_gib']<256 for t in tasks)
    assert all(t['bond_dim']==(3 if t['model']=='fermi_hubbard' else 4) for t in tasks)
    for case in {t['case'] for t in tasks}:
        assert len({t['initial_state']['tensor_hash'] for t in tasks if t['case']==case})==1


def test_real_copied_workers_load_identical_tensors(bundle,tmp_path):
    common=load(bundle/'source/pyqed/_letta_one_site_opt/benchmarks/2D/cluster_sept27/run_compression.py','rectangle_smoke')
    common.CASES=(('bose_hubbard',(2,2),2),)
    planfile=tmp_path/'run/plan.json'
    common.make_plan(SimpleNamespace(root=bundle,output=planfile,seeds=[731],models=None,cases=None,
        algorithms=['one-site','cbe','two-site'],profiles=['als4-lsmr400','als40-lsmr400'],
        sweeps=2,nonlinear_iterations=10,workspace_mb=128,memory_gib=64,cpus=1))
    spool=tmp_path/'spool'; spool.mkdir()
    copied=spool/'slurm_script'; shutil.copyfile(bundle/'submit_rectangles.sh',copied)
    for i in range(5):
        result=subprocess.run(['bash',str(copied),'worker'],cwd=spool,
            env=env(bundle,RUN_DIR=str(planfile.parent),SLURM_ARRAY_TASK_ID=str(i)),
            capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    reports=[json.loads(p.read_text()) for p in planfile.parent.glob('results/*/*.json')]
    assert len(reports)==5 and all(r['status']=='completed' for r in reports)
    assert len({r['initial_tensor_hash'] for r in reports})==1
    assert all(r['initial_tensor_hash']==r['task']['initial_state']['tensor_hash'] for r in reports)
    assert all(abs(r['physical_energy']-r['energy'])<1e-8 for r in reports)


def test_missing_frozen_seed_rejected_before_submission(bundle,tmp_path):
    result=subprocess.run(['bash',str(bundle/'submit_rectangles.sh'),'plan'],
        env=env(bundle,RUN_DIR=str(tmp_path/'run'),SEEDS='999'),capture_output=True,text=True)
    assert result.returncode!=0 and 'No frozen initial state' in result.stderr
    assert not (tmp_path/'run/plan.json').exists()
