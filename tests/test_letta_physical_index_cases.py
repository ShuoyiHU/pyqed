"""Audit LETTA physical-index incidences against independent physical states.

Reference: _letta_one_site_opt/letta_cbe_physical_index_cases.pdf.
Every nonempty subset of L,A,B,R is tested with every possible home site,
plus mixed forward ties, in singlet and doublet sectors. Dense determinant
operators are validation references only; the tested solver stays reduced.
"""
import itertools
from collections import Counter
import numpy as np
import pytest
from pyqed._letta_one_site_opt import ReducedLatticeLETTA, LETTADMROptions
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.reduced_solver import ReducedSweepContext, _energy
from pyqed._letta_two_site_opt.reduced_solver import reduced_pair_problem, _pair_vector_from_sources
from pyqed._letta_two_site_opt.reduced_compression import ReducedPairMetricRoot
from test_letta_qchem import determinant_hamiltonian
CASES = [(s, home) for n in range(1, 5) for s in itertools.combinations(range(4), n) for home in s] + [('mixed', None)]

@pytest.mark.parametrize('incidence', CASES, ids=lambda s: str(s))
@pytest.mark.parametrize('nelec', [(2, 2), (2, 1)])
def test_category_actions_and_cache_against_physical_state(incidence, nelec):
    h = np.diag([0.1, 0.2, 0.3, 0.4])
    h += np.array([[0, -1, 0.13, 0.21], [-1, 0, -0.7, 0.17], [0.13, -0.7, 0, -0.9], [0.21, 0.17, -0.9, 0]])
    g = np.zeros((4,) * 4)
    for i in range(4):
        g[i, i, i, i] = 4
    p = ElectronicProblem(h, g, nelec)
    ham = p.su2_mpo()
    dense = determinant_hamiltonian(p)
    incidence, home = incidence
    nb = [{i} for i in range(4)]
    if incidence == 'mixed':
        nb = [set(range(i, 4)) for i in range(4)]
    else:
        for i in incidence:
            nb[i].add(home)
    s = ReducedLatticeLETTA.random((1, 4), symmetry=p.symmetry('su2'), neighborhoods=tuple(((i,) + tuple(sorted(x - {i})) for i, x in enumerate(nb))), multiplets_per_sector=1, seed=37, real=False)
    opts = LETTADMROptions(matrix_free=True, dense_solver_threshold=0)
    ctx = ReducedSweepContext(s, ham)
    rng = np.random.default_rng(77)
    factor = s.target_two_j + 1
    for site in (1, 2):
        emb = ctx.embeddings[site]
        s.tensors[site] = emb.unpack_source(rng.normal(size=emb.source_size) + 1j * rng.normal(size=emb.source_size))
        ctx.synchronize([site])
    v = s.state_vector()
    n = factor * np.vdot(v, v)
    energy = np.vdot(v, dense @ v) / np.vdot(v, v)
    assert _energy(s, ham, stable=True) == pytest.approx(energy.real, abs=2e-10)
    for site in (1, 2):
        prob = ctx.local_problem(site, opts)
        x = prob.embedding.pack_source(s.tensors[site])
        np.testing.assert_allclose(np.vdot(x, prob.apply_metric(x)), n, rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(np.vdot(x, prob.apply_hamiltonian(x)), n * energy, rtol=2e-10, atol=2e-10)
    pair = reduced_pair_problem(s, ham, 1, matrix_free=True, dense_solver_threshold=0)
    root = ReducedPairMetricRoot(pair, s, 1e-12)
    x = pair.old_vector
    np.testing.assert_allclose(np.vdot(x, pair.apply_metric(x)), n, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(np.vdot(x, pair.apply_hamiltonian(x)), n * energy, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(np.vdot(root.apply(x), root.apply(x)), n, rtol=2e-10, atol=2e-10)
    from pyqed._letta_one_site_opt.reduced_cbe import _canonical_pair_residual
    expected = root.unwhiten_adjoint(pair.apply_hamiltonian(x) - energy * pair.apply_metric(x))
    np.testing.assert_allclose(_canonical_pair_residual(pair, s, ham, root), expected, atol=2e-09, rtol=2e-09)
    from pyqed._letta_one_site_opt.reduced_cbe import _tangent_coordinates
    from pyqed._letta_one_site_opt.tangent_projection import _factor_tangent_matrix, project_tangent
    from pyqed._letta_two_site_opt.reduced_compression import _physical_factor_map
    le, re = (pair.frontier.site_embedding(s, k) for k in (1, 2))
    a, b = (le.pack_source(s.tensors[1]), re.pack_source(s.tensors[2]))
    maps = _tangent_coordinates(s, pair, (le, re), Counter(s.bond_sectors[1]), opts)
    matrices = []
    for side, c in enumerate(maps):
        mat = _factor_tangent_matrix(c, root, 1, side)
        matrices.append(mat)
        z = rng.normal(size=c.rank) + 1j * rng.normal(size=c.rank)
        delta = c.expand(z)
        direct = root.apply(_pair_vector_from_sources(pair.layout, le, re, delta, b) if side == 0 else _pair_vector_from_sources(pair.layout, le, re, a, delta))
        np.testing.assert_allclose(mat @ z, direct, atol=2e-09, rtol=2e-09)
        forward, reverse = _physical_factor_map(c, root, 1, side)
        np.testing.assert_allclose(root.project_physical(forward(z)), direct, atol=2e-09, rtol=2e-09)
        w = rng.normal(size=root.size) + 1j * rng.normal(size=root.size)
        np.testing.assert_allclose(np.vdot(w, direct), np.vdot(reverse(root.embed_physical(w)), z), atol=2e-09, rtol=2e-09)
    matrix = np.concatenate(matrices, axis=1)
    gradient = rng.normal(size=root.size) + 1j * rng.normal(size=root.size)
    missing, report = project_tangent(maps, root, 1, gradient, workspace_mb=256, rank_tolerance=1e-12, normal_tolerance=1e-09)
    assert missing is not None, report
    reference = gradient - matrix @ np.linalg.lstsq(matrix, gradient, rcond=1e-12)[0]
    np.testing.assert_allclose(missing, reference, atol=2e-09, rtol=2e-09)
