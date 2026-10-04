"""Large manifests and memory checks must not allocate large tensor networks."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "pyqed/_letta_one_site_opt/benchmarks/2D/cluster_sept18/run_large_2d.py"
SPEC = importlib.util.spec_from_file_location("letta_cluster", PATH)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


@pytest.mark.parametrize('qos', [None, 'large', 'huge', 'default', ''])
def test_launcher_qos_override_preserves_plan_and_unlimited_array(tmp_path, qos):
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'tasks': [{}] * 624}))
    original = plan.read_bytes()
    executable = tmp_path / 'sbatch'
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\necho 123456\n')
    executable.chmod(0o755)
    capture = tmp_path / 'args.txt'
    env = dict(os.environ, PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],
               CAPTURE_ARGS=str(capture), PLAN=str(plan), PARTITION='gubing',
               RUN_ROOT=str(PATH.parent), PYQED_REPO=str(PATH.parents[5]),
               LETTA_PYTHON=sys.executable, MPLCONFIGDIR=str(tmp_path / 'mpl'))
    if qos is None:
        env.pop('QOS', None)
    else:
        env['QOS'] = qos
    result = subprocess.run(['bash', str(PATH.with_name('submit_large_2d.sh')), 'submit'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = capture.read_text().splitlines()
    assert args[args.index('-p')+1] == 'gubing'
    assert '--time=0' in args and '--array=0-623' in args
    if qos in ('', 'default'):
        assert '-q' not in args
    else:
        assert args[args.index('-q')+1] == ('huge' if qos is None else qos)
    assert plan.read_bytes() == original


def test_bundle_preflight_without_editable_install_fallback(tmp_path):
    repo = PATH.parents[5]
    stage = tmp_path / "bundle"
    subprocess.run([sys.executable, str(PATH.with_name('stage_bundle.py')),
                    '--repo', str(repo), '--output', str(stage)], check=True)
    assert (stage / 'pyqed/davidson.py').is_file()
    # -I -S suppresses cwd, PYTHONPATH and .pth editable-install hooks. Add
    # dependency directories explicitly, without executing their .pth files.
    libraries = [p for p in sys.path if p and Path(p).name in {'site-packages', 'dist-packages'}]
    bootstrap = (
        'import sys,runpy; sys.path.extend(' + repr(libraries) + '); '
        'sys.argv=[sys.argv[1], "preflight", "--repo", sys.argv[2]]; '
        'runpy.run_path(sys.argv[0],run_name="__main__")'
    )
    result = subprocess.run([sys.executable, '-I', '-S', '-W', 'ignore', '-c', bootstrap,
                             str(stage / PATH.relative_to(repo)), str(stage)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert str(stage / 'pyqed/__init__.py') in result.stdout


def test_runtime_rejects_submodules_from_other_checkouts(tmp_path, monkeypatch):
    import types
    foreign = types.ModuleType('pyqed.foreign_bundle_probe')
    foreign.__file__ = str(tmp_path / 'foreign.py')
    monkeypatch.setitem(sys.modules, foreign.__name__, foreign)
    with pytest.raises(ImportError, match='imports escaped'):
        driver.load_runtime(PATH.parents[5])


def test_default_manifest_is_complete_and_rotates_rectangles(tmp_path):
    plan = tmp_path / "plan.json"
    driver.main(["plan", "--output", str(plan)])
    tasks = json.loads(plan.read_text())["tasks"]
    assert len(tasks) == 624
    assert len({(t['case'], t['solver']) for t in tasks}) == 624
    assert {tuple(t['requested_shape']) for t in tasks} == {
        tuple(map(int, s.split('x'))) for s in driver.SHAPES.split()}
    assert all(t['shape'] == [9, 3] for t in tasks if t['requested_shape'] == [3, 9])
    assert all(t['max_sweeps'] == 100 for t in tasks)
    with pytest.raises(FileExistsError):
        driver.main(["plan", "--output", str(plan)])


@pytest.mark.parametrize('model', driver.MODELS)
def test_memory_count_matches_production_frontier_shapes(model):
    from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
    from pyqed._letta_one_site_opt import LatticeLETTA, LETTAEnvironmentCache
    h = build_model(model, dimension='2d', size=(3, 2))
    state = LatticeLETTA.random((3, 2), physical_dim=h.physical_dim, bond_dim=2)
    cache = LETTAEnvironmentCache(state, h.mpo)
    largest = max(np.prod([cache.label_dimensions[z] for z in frontier]) for frontier in cache.frontiers) * 8
    estimate = driver.memory_estimate(model, (3, 2), 2)
    assert estimate['largest_h_environment_gib'] == largest / 1024**3


def test_memory_block_is_recorded_before_loading_runtime(tmp_path, monkeypatch):
    plan = tmp_path / 'plan.json'
    driver.main(['plan', '--output', str(plan), '--shapes', '9x9', '--models', 'fermi_hubbard',
                 '--solvers', 'one_site', '--bond-dims', '8', '--seeds', '731'])
    def forbidden(*args):
        raise AssertionError('large job loaded its runtime before the memory check')
    monkeypatch.setattr(driver, 'load_runtime', forbidden)
    assert driver.main(['run', '--plan', str(plan), '--task-index', '0']) == 2
    output = next((tmp_path / 'results').rglob('*.json'))
    assert json.loads(output.read_text())['status'] == 'resource_blocked'
    driver.main(['collect', '--plan', str(plan)])
    assert 'resource_blocked' in (tmp_path / 'summary.csv').read_text()


def test_separate_two_site_cap_and_omission(tmp_path):
    path = tmp_path / 'plan.json'
    driver.main(['plan', '--output', str(path), '--shapes', '3x3', '--models', 'ising',
                 '--bond-dims', '4', '--seeds', '731', '--max-sweeps', '60', '--two-site-max-sweeps', '7'])
    tasks = json.loads(path.read_text())['tasks']
    assert [t['max_sweeps'] for t in tasks] == [60, 60, 7]
    second = tmp_path / 'without.json'
    driver.main(['plan', '--output', str(second), '--solvers', 'one_site', 'cbe'])
    assert {t['solver'] for t in json.loads(second.read_text())['tasks']} == {'one_site','cbe'}
