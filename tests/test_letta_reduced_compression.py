"""Physical-metric, gauge and budget checks of reduced factor compression."""
import numpy as np
import pytest

from pyqed._letta_compression import MetricCompressionOptions, _balance
from pyqed._letta_one_site_opt import ReducedLatticeLETTA
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_two_site_opt.reduced_solver import reduced_pair_problem
from pyqed._letta_two_site_opt.reduced_compression import (
    ReducedPairMetricRoot, reduced_factor_blocks, compress_reduced_pair,
)
from test_letta_qchem import integrals


def example(ties=None):
    p = ElectronicProblem(*integrals(3), (2, 1))
    s = ReducedLatticeLETTA.random((1, 3), symmetry=p.symmetry('su2'),
        neighborhoods=ties, multiplets_per_sector=1, real=False, seed=87)
    pair = reduced_pair_problem(s, p.su2_mpo(), 1, matrix_free=True, dense_solver_threshold=0)
    le, re = (pair.frontier.site_embedding(s, i) for i in (1, 2))
    a, b = le.pack_source(s.tensors[1]), re.pack_source(s.tensors[2])
    from collections import Counter
    return s, pair, le, re, a, b, Counter(s.bond_sectors[1])


@pytest.mark.parametrize('ties', [None, ((0,), (1,), (2,)), ((0, 2), (1, 0), (2, 1))])
def test_reduced_square_root_has_exact_metric_and_adjoint(ties):
    s, p, *_ = example(ties)
    root = ReducedPairMetricRoot(p, s, 1e-12)
    rng = np.random.default_rng(19)
    x = rng.normal(size=p.local_dimension)+1j*rng.normal(size=p.local_dimension)
    y = rng.normal(size=root.size)+1j*rng.normal(size=root.size)
    np.testing.assert_allclose(root.adjoint(root.apply(x)), p.apply_metric(x), atol=2e-11)
    np.testing.assert_allclose(np.vdot(y, root.apply(x)), np.vdot(root.adjoint(y), x), atol=2e-11)
    assert p.metric is None


@pytest.mark.parametrize('ties', [None, ((0, 2), (1, 0), (2, 1))])
def test_reduced_factor_gauge_blocks_preserve_merged_state(ties):
    from pyqed._letta_two_site_opt.reduced_solver import _pair_vector_from_sources
    s, p, le, re, a, b, retained = example(ties)
    groups = reduced_factor_blocks(s, p.left_site, le, re, retained)
    used_a = np.concatenate([x.ravel() for x, _ in groups])
    used_b = np.concatenate([x.ravel() for _, x in groups])
    assert len(set(used_a)) == len(used_a)
    assert len(set(used_b)) == len(used_b)
    aa, bb = _balance(a, b, groups)
    merge = lambda x, y: _pair_vector_from_sources(p.layout, le, re, x, y)
    np.testing.assert_allclose(merge(aa, bb), merge(a, b), atol=2e-12)
    invisible_a, invisible_b = a.copy(), b.copy()
    invisible_a[used_a] = 0
    invisible_b[used_b] = 0
    np.testing.assert_allclose(merge(invisible_a, b), 0., atol=1e-14)
    np.testing.assert_allclose(merge(a, invisible_b), 0., atol=1e-14)


@pytest.mark.parametrize('solver', ['als', 'variable-projection', 'joint-ls', 'grassmann-newton'])
def test_all_reduced_compression_solvers_use_physical_metric(solver, monkeypatch):
    s, p, le, re, a, b, retained = example()
    rng = np.random.default_rng(5)
    target = p.old_vector+.03*(rng.normal(size=p.local_dimension)+1j*rng.normal(size=p.local_dimension))
    def forbidden(*args, **kwargs):
        raise AssertionError('compression used determinant-space reconstruction')
    monkeypatch.setattr(ReducedLatticeLETTA, 'state_vector', forbidden)
    fit = compress_reduced_pair(target, p, s, a, b, retained,
        options=MetricCompressionOptions(solver=solver, max_iterations=20,
            als_max_iterations=7, lsmr_max_iterations=25))
    from pyqed._letta_two_site_opt.reduced_solver import _pair_vector_from_sources
    error = _pair_vector_from_sources(p.layout, le, re, fit.left, fit.right)-target
    expected = np.vdot(error, p.apply_metric(error)).real
    assert fit.loss == pytest.approx(expected, abs=1e-11)
    assert fit.loss <= fit.diagnostics['initial_loss']+1e-11
    assert fit.diagnostics['requested_solver'] == solver
    assert fit.diagnostics['used_solver'] == solver


def test_reduced_als_and_linear_budgets_are_independent():
    s, p, le, re, a, b, retained = example()
    rng = np.random.default_rng(9)
    target = p.old_vector+.2*(rng.normal(size=p.local_dimension)+1j*rng.normal(size=p.local_dimension))
    fit = compress_reduced_pair(target, p, s, a, b, retained,
        options=MetricCompressionOptions(als_max_iterations=1, lsmr_max_iterations=1))
    assert fit.iterations == 1
    reports = fit.diagnostics['linear_solves']
    assert len(reports) == 2
    assert all(r['max_iterations'] == 1 and r['iterations'] <= 1 for r in reports)
    assert not fit.diagnostics['optimizer_success']
    assert fit.diagnostics['status'] != 'converged'


@pytest.mark.parametrize('solver', ['als', 'variable-projection', 'joint-ls', 'grassmann-newton'])
def test_reduced_two_site_dispatches_requested_compressor(solver):
    from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg
    from pyqed._letta_one_site_opt.reduced_solver import _energy
    p = ElectronicProblem(*integrals(3), (2, 1))
    s = ReducedLatticeLETTA.random((1, 3), symmetry=p.symmetry('su2'), seed=87)
    h = p.su2_mpo()
    before = _energy(s, h, stable=True)
    result = letta_two_site_dmrg(h, state=s, bond_dim=3,
        options=LETTATwoSiteOptions(max_sweeps=1, split_method='conditional-svd',
            compression=MetricCompressionOptions(solver=solver, max_iterations=10,
                als_max_iterations=2, lsmr_max_iterations=12)))
    assert result.energy <= before+1e-10
    assert result.state.symmetry_violation() == 0.
    for update in result.history[0].updates:
        assert update.compression_diagnostics['requested_solver'] == solver
        assert update.compression_diagnostics['used_solver'] == solver
        if solver == 'als':
            assert update.truncation_iterations <= 2
            assert all(x['iterations'] <= 12 for x in update.compression_diagnostics['linear_solves'])
