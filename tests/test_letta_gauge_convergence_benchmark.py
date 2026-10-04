"""Numerical accounting checks for the directly runnable gauge benchmark."""
import csv
import json

import numpy as np
import pytest

from pyqed._letta_one_site_opt.benchmarks.run_gauge_convergence import DEFAULTS, run_comparison, save_report


@pytest.mark.parametrize('exact_limit', [0, 32])
def test_report_matches_physical_energy_and_solver_stopping_rule(tmp_path, exact_limit):
    config = dict(DEFAULTS, columns=4, bond_dims=(2,), repeats=1,
                  max_sweeps=4, exact_max_dimension=exact_limit, output=tmp_path)
    report = run_comparison(config)
    assert (report['exact_energy'] is None) == (exact_limit == 0)
    assert len({r['initial_hash'] for r in report['runs']}) == 1
    for run in report['runs']:
        assert abs(run['energy'] - run['physical_energy']) < 1e-10
        previous = run['initial_energy']
        for step in run['history']:
            assert step['energy_density_change'] == pytest.approx(abs(step['energy'] - previous) / 4, abs=1e-12)
            assert step['elapsed_seconds'] >= 0
            assert step['pass_seconds'] >= 0
            previous = step['energy']
        assert run['converged'] == (run['history'][-1]['energy_density_change'] <= config['tolerance'])
        if exact_limit:
            assert abs(run['energy_error']) < 1e-8
            assert run['target_sweep'] is not None
            assert np.isfinite(run['physical_residual'])
        else:
            assert run['target_seconds'] is None
        if run['gauge'] == 'frontier':
            assert run['canonical_metric_hits'] == 4 * run['sweeps']
    save_report(report, tmp_path, plot=False)
    loaded = json.loads((tmp_path / 'results.json').read_text())
    assert len(loaded['runs']) == 2
    with (tmp_path / 'sweeps.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == sum(r['sweeps'] for r in report['runs'])
