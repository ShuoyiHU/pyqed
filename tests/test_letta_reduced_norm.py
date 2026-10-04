"""Native norm recursions against magnetic-component contractions."""
import numpy as np
import pytest

from pyqed._letta_one_site_opt import ReducedLatticeLETTA
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.orbital_ordering import tie_neighborhoods
from pyqed._letta_one_site_opt.reduced_frontier import ReducedFrontier
from pyqed._letta_one_site_opt.reduced_norm import ReducedNormChain
from pyqed._letta_one_site_opt.reduced_contraction import (
    CanonicalEnvironmentChain, identity_canonical_factors, expand_reduced_mps_site,
    reduce_expanded_mps_site, _component_axis_layout,
)


@pytest.mark.parametrize('n,nelec,two_s', [(1, (1, 0), 1), (3, (2, 1), 1),
                                        (4, (2, 2), 0), (3, (2, 0), 2)])
@pytest.mark.parametrize('ties', ['none', 'nn', 'carried'])
def test_native_norm_boundaries_and_adjoint_match_components(n, nelec, two_s, ties):
    p = ElectronicProblem(np.eye(n), np.zeros((n,)*4), nelec)
    neighborhoods = (tuple((i,) for i in range(n)) if ties == 'none'
        else tie_neighborhoods(n, [(0, n-1)] if ties == 'carried' and n > 1 else [],
                               nearest=True, carry=ties == 'carried'))
    s = ReducedLatticeLETTA.random((1, n), symmetry=p.symmetry('su2', two_s=two_s),
        neighborhoods=neighborhoods, multiplets_per_sector=2, real=False, seed=126)
    sites = tuple(ReducedFrontier.from_state(s).to_mps(s))
    ref = CanonicalEnvironmentChain.build(sites, identity_canonical_factors(sites))
    chain = ReducedNormChain.build(sites)
    np.testing.assert_allclose(chain.expectation(), ref.expectation(), atol=3e-12)
    rng = np.random.default_rng(57)
    for i, a in enumerate(sites):
        for side, axis, cut in [('left', 0, i), ('right', 2, i+1)]:
            offsets, mults, size = _component_axis_layout(a, axis)
            expanded = np.zeros((size, size), complex)
            blocks = getattr(chain, side)[cut]
            scale = np.exp(getattr(chain, side+'_log_scales')[cut])
            for q, g in blocks.items():
                d = g.shape[0]
                assert d == mults[q]
                # q carries an SU2 irrep, even for U(1) x SU(2).
                from pyqed._letta_one_site_opt.reduced_symmetry import _sector_irrep
                value = np.kron(g, np.eye(_sector_irrep(q).dim))*scale
                start = offsets[q]
                expanded[start:start+len(value), start:start+len(value)] = value
            expected = getattr(ref, side)[i][0]*np.exp(getattr(ref, side+'_log_scales')[i])
            np.testing.assert_allclose(expanded, expected, atol=3e-12)
        trial = a.copy()
        trial.data = {key: rng.normal(size=b.shape)+1j*rng.normal(size=b.shape)
                      for key, b in a.data.items()}
        expected = reduce_expanded_mps_site(a, ref.local_action(i, expand_reduced_mps_site(trial)))
        actual = chain.local_action(i, trial.data)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], atol=3e-12)


def test_norm_and_frontier_gauge_do_not_expand_magnetic_components(monkeypatch):
    import pyqed._letta_one_site_opt.reduced_contraction as components
    from pyqed._letta_one_site_opt.reduced_gauge import canonicalize_reduced_frontier
    p = ElectronicProblem(np.eye(3), np.zeros((3,)*4), (2, 1))
    s = ReducedLatticeLETTA.random((1, 3), symmetry=p.symmetry('su2'), real=False, seed=91)
    original = s.state_vector()
    def forbidden(*args, **kwargs):
        raise AssertionError('magnetic MPS expansion in native norm/gauge')
    monkeypatch.setattr(components, 'expand_reduced_mps_site', forbidden)
    monkeypatch.setattr(ReducedLatticeLETTA, 'state_vector', forbidden)
    np.testing.assert_allclose(s.norm(), np.vdot(original, original), atol=1e-12)
    canonicalize_reduced_frontier(s, 1)
    np.testing.assert_allclose(s.norm(), np.vdot(original, original), atol=1e-12)
    monkeypatch.undo()
    np.testing.assert_allclose(s.state_vector(), original, atol=1e-12)


@pytest.mark.parametrize('ties', ['none', 'nn', 'carried'])
def test_conditional_support_projector_preserves_metric_action(ties):
    p = ElectronicProblem(np.eye(3), np.zeros((3,)*4), (2, 1))
    neighborhoods = (tuple((i,) for i in range(3)) if ties == 'none' else
        tie_neighborhoods(3, [(0, 2)] if ties == 'carried' else [],
                          nearest=True, carry=ties == 'carried'))
    s = ReducedLatticeLETTA.random((1, 3), symmetry=p.symmetry('su2'), real=False,
        neighborhoods=neighborhoods, multiplets_per_sector=3, seed=33)
    frontier = ReducedFrontier.from_state(s)
    chain = ReducedNormChain.build(frontier.to_mps(s))
    rng = np.random.default_rng(289)
    for i in range(3):
        embedding = frontier.site_embedding(s, i)
        projector = chain.local_projector(i, embedding, 1e-12)
        assert projector is not None
        x = rng.normal(size=embedding.source_size)+1j*rng.normal(size=embedding.source_size)
        px = projector(x)
        def action(v):
            return embedding.adjoint(embedding.pack_target(chain.local_action(i,
                embedding.unpack_target(embedding.apply(v)))))
        np.testing.assert_allclose(action(px), action(x), atol=2e-11)
        np.testing.assert_allclose(projector(px), px, atol=2e-12)
        assert np.linalg.norm(px) <= np.linalg.norm(x)+1e-12
