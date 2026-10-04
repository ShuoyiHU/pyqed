"""Run the new 2D entry point with and without the expensive two-site path."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
from pyqed._letta_one_site_opt.benchmarks import condensed_runner as runner

PATH = Path(__file__).resolve().parents[1] / 'pyqed/_letta_one_site_opt/benchmarks/2D/compare.py'
spec = importlib.util.spec_from_file_location('letta_2d_compare', PATH)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def test_initialization_does_not_expand_the_hilbert_space(monkeypatch):
    def forbidden(*args):
        raise AssertionError('dense state expansion during initialization')
    monkeypatch.setattr(runner, 'mps_state_vector', forbidden)
    model = build_model('ising', dimension='2d', size=(2, 3))
    ties = compare.neighborhoods_for_shape(2, 3, 'bidirectional')
    initial = runner.make_shared_initial_state(model, bond_dim=2, neighborhoods=ties)
    assert initial.letta.neighborhoods == ties
    np.testing.assert_allclose(initial.letta.norm(), 1., atol=1e-12)


@pytest.mark.parametrize('model', compare.MODELS)
def test_2d_comparison_can_disable_two_site(model, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('two-site solve was not disabled')
    monkeypatch.setattr(runner, 'letta_two_site_dmrg', forbidden)
    output = tmp_path / f'{model}.json'
    report = compare.main(['--max-sweeps', '1', '--bond-dim', '1', '--skip-two-site',
                           '--exact-max-dimension', '0', '--output', str(output)], model=model)
    assert [r['solver'] for r in report['records']] == ['letta_one_site', 'letta_cbe_strict']
    assert not report['solver_failures']
    assert json.loads(output.read_text())['records'][0]['energy'] == report['records'][0]['energy']


def test_two_site_has_a_separate_cap_and_custom_ties(tmp_path):
    ties = [[0, 2, 3], [1, 3], [2, 0], [3, 1]]
    path = tmp_path / 'ties.json'
    path.write_text(json.dumps(ties))
    report = compare.main(['--max-sweeps', '2', '--two-site-max-sweeps', '1',
                           '--bond-dim', '1', '--ties-json', str(path),
                           '--exact-max-dimension', '0'])
    assert report['neighborhoods'] == ties
    assert report['two_site_max_sweeps'] == 1
    assert [r['solver'] for r in report['records']] == ['letta_one_site', 'letta_cbe_strict', 'letta_two_site']
    assert report['records'][-1]['sweeps'] == 1
    assert len({r['initial_state_fingerprint'] for r in report['records']}) == 1
