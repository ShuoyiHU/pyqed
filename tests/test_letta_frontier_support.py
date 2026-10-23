"""Exact support pruning: independent small-system paths and dense references."""
from collections import Counter
from itertools import product

import numpy as np
import pytest

from pyqed._letta_one_site_opt import (
    LETTADMROptions, ReducedFrontier, ReducedLatticeLETTA,
    ReducedPhysicalBasis, ReducedSymmetry, reduced_mps_state_vector,
)
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.reduced_solver import _native_local_problem, _solve_local_problem
from test_letta_qchem import determinant_hamiltonian


def state(neighborhoods=((0, 1, 2), (1, 2), (2,)), *, copies=1):
    symmetry = ReducedSymmetry.su2(ReducedPhysicalBasis.spatial_orbital(),
                                  target_charge=3, target_two_j=1)
    return ReducedLatticeLETTA.random((1, 3), symmetry=symmetry,
        neighborhoods=neighborhoods, multiplets_per_sector=copies,
        seed=2027, real=False, normalize=False)


def complete_paths(s):
    """Enumerate sector paths without consulting frontier code or coefficients."""
    physical = s.physical_basis.reduced_states
    for configuration in product(range(len(physical)), repeat=s.nsites):
        paths = [(s.symmetry.identity,)]
        for i, p in enumerate(configuration):
            paths = [path + (right,) for path in paths
                     for left, own, right in s.tensors[i]
                     if left == path[-1] and own == physical[p].sector]
        for path in paths:
            if path[-1] == s.symmetry.sector:
                yield configuration, path


def source_support(s, site, embedding):
    physical = s.physical_basis.reduced_states
    supported = set()
    grids = {key: np.arange(lo, hi).reshape(embedding.source_layout.shapes[key])
             for key, (lo, hi) in embedding.source_layout.offsets.items()}
    for configuration, path in complete_paths(s):
        own = physical[configuration[site]]
        key = (path[site], own.sector, path[site + 1])
        dependencies = tuple(configuration[j] for j in s.neighborhoods[site][1:])
        supported.update(grids[key][(slice(None), own.copy) + dependencies +
                                   (slice(None),)].ravel().tolist())
    return np.array(sorted(supported), dtype=int)


def restricted_bond(s, cut, keep):
    bonds = list(s.bond_sectors)
    bonds[cut] = tuple(q for q in bonds[cut] if keep(q))
    tensors = [{key: a.copy() for key, a in blocks.items()
                if (i != cut or key[2] in bonds[cut]) and
                   (i != cut + 1 or key[0] in bonds[cut])}
               for i, blocks in enumerate(s.tensors)]
    return ReducedLatticeLETTA(s.lattice_shape, s.symmetry, tensors,
        bond_sectors=bonds, neighborhoods=s.neighborhoods, normalize=False)


@pytest.mark.parametrize('copies', [1, 2])
def test_three_electrons_cannot_condition_on_two_doubly_occupied_sites(copies):
    s = state(copies=copies)
    frontier = ReducedFrontier.from_state(s, prune=True)
    embedding = frontier.site_embedding(s, 0)
    expected = source_support(s, 0, embedding)
    np.testing.assert_array_equal(embedding.supported_source_indices, expected)
    assert 0 < embedding.source_size < embedding.source_layout.size
    packed = embedding.pack_source(s.tensors[0])
    restored = embedding.unpack_source(packed)
    for key, block in restored.items():
        np.testing.assert_array_equal(block[:, :, 2, 2, :], 0.)
    raw = embedding.source_layout.pack(restored)
    np.testing.assert_array_equal(raw[expected],
        embedding.source_layout.pack(s.tensors[0])[expected])
    np.testing.assert_array_equal(np.flatnonzero(raw), expected)


@pytest.mark.parametrize('neighborhoods', [
    ((0, 1, 2), (1, 2), (2,)),
    ((0, 2), (1, 0), (2, 1)),
    ((0,), (1, 0, 2), (2, 0)),
])
@pytest.mark.parametrize('compact', [False, True])
def test_pruned_frontier_preserves_complex_wavefunction_and_adjoint(neighborhoods, compact):
    s = state(neighborhoods)
    frontier = ReducedFrontier.from_state(s, prune=True)
    old = ReducedFrontier.from_state(s, prune=False)
    reference = s.state_vector()
    for model in (frontier, old):
        vector = reduced_mps_state_vector(model.to_mps(s, compact=compact),
            s.physical_basis, target_sector=s.symmetry.sector)
        np.testing.assert_allclose(vector, reference, atol=2e-13, rtol=2e-13)
    rng = np.random.default_rng(31)
    for i in range(s.nsites):
        emb = frontier.site_embedding(s, i)
        np.testing.assert_array_equal(emb.supported_source_indices, source_support(s, i, emb))
        x = rng.normal(size=emb.source_size) + 1j * rng.normal(size=emb.source_size)
        y = rng.normal(size=emb.target_size) + 1j * rng.normal(size=emb.target_size)
        np.testing.assert_allclose(np.vdot(emb.apply(x), y), np.vdot(x, emb.adjoint(y)), atol=2e-13)
        np.testing.assert_allclose(emb.pack_source(emb.unpack_source(x)), x, atol=0.)


def test_fixed_intermediate_charge_sector_prunes_more_than_total_charge():
    s = state()
    constrained = restricted_bond(s, 1, lambda q: q.charge == 1)
    full = ReducedFrontier.from_state(s, prune=True).site_embedding(s, 0)
    frontier = ReducedFrontier.from_state(constrained, prune=True)
    emb = frontier.site_embedding(constrained, 0)
    assert emb.source_size < full.source_size
    np.testing.assert_array_equal(emb.supported_source_indices, source_support(constrained, 0, emb))
    allowed = set(emb.supported_source_indices)
    for key, (lo, hi) in emb.source_layout.offsets.items():
        grid = np.arange(lo, hi).reshape(emb.source_layout.shapes[key])
        for second in (0, 1):
            assert allowed.isdisjoint(grid[:, :, :, second, :].ravel())
    mps = frontier.to_mps(constrained)
    for i in range(constrained.nsites - 1):
        contexts = {(path[i + 1], tuple(config[j] for j in frontier.right_variables(i)))
                    for config, path in complete_paths(constrained)}
        multiplicities = Counter(constrained.bond_sectors[i])
        assert len(mps[i].qns[2]) == sum(multiplicities[q] for q, _ in contexts)



def test_fixed_intermediate_spin_sector_rejects_charge_allowed_assignments():
    symmetry = ReducedSymmetry.su2(ReducedPhysicalBasis.spatial_orbital(),
                                  target_charge=4, target_two_j=0)
    s = ReducedLatticeLETTA.random((2, 2), symmetry=symmetry,
        neighborhoods=((0, 1, 2, 3), (1, 2, 3), (2, 3), (3,)),
        seed=2029, real=False, normalize=False)
    constrained = restricted_bond(s, 1, lambda q: q.charge == 2 and q.irrep.two_j == 2)
    frontier = ReducedFrontier.from_state(constrained, prune=True)
    emb = frontier.site_embedding(constrained, 0)
    np.testing.assert_array_equal(emb.supported_source_indices, source_support(constrained, 0, emb))
    restored = emb.unpack_source(np.ones(emb.source_size))
    for key, block in restored.items():
        if key[1].charge != 1:
            np.testing.assert_array_equal(block, 0.)
        else:
            np.testing.assert_array_equal(block[:, :, 0], 0.)
            np.testing.assert_array_equal(block[:, :, 2], 0.)
    np.testing.assert_allclose(reduced_mps_state_vector(frontier.to_mps(constrained),
        constrained.physical_basis, target_sector=symmetry.sector),
        constrained.state_vector(), atol=2e-13)

def test_support_depends_on_allowed_blocks_not_accidental_tensor_zeros():
    s = state()
    frontier = ReducedFrontier.from_state(s, prune=True)
    before = frontier.site_embedding(s, 0)
    changed = s.copy()
    for blocks in changed.tensors:
        for block in blocks.values():
            block[...] = 0.
    after = frontier.site_embedding(changed, 0)
    np.testing.assert_array_equal(after.supported_source_indices, before.supported_source_indices)
    fresh = ReducedFrontier.from_state(changed, prune=True).site_embedding(changed, 0)
    np.testing.assert_array_equal(fresh.supported_source_indices, before.supported_source_indices)
    assert after.source_size == before.source_size


def test_reused_frontier_cannot_keep_support_from_old_sector_layout():
    s = state()
    frontier = ReducedFrontier.from_state(s, prune=True)
    before = frontier.site_embedding(s, 0)
    changed = restricted_bond(s, 1, lambda q: q.charge == 1)
    after = frontier.site_embedding(changed, 0)
    fresh = ReducedFrontier.from_state(changed, prune=True).site_embedding(changed, 0)
    assert after.source_size < before.source_size
    np.testing.assert_array_equal(after.supported_source_indices, fresh.supported_source_indices)
    np.testing.assert_array_equal(after.target_indices, fresh.target_indices)
    np.testing.assert_allclose(after.apply(after.pack_source(changed.tensors[0])),
                               fresh.apply(fresh.pack_source(changed.tensors[0])))


def test_reused_frontier_rebuilds_or_rejects_changed_tie_topology():
    s = state()
    frontier = ReducedFrontier.from_state(s, prune=True)
    frontier.site_embedding(s, 0)
    changed = state(((0, 2, 1), (1, 0), (2, 0)))
    fresh = ReducedFrontier.from_state(changed, prune=True).site_embedding(changed, 0)
    try:
        after = frontier.site_embedding(changed, 0)
    except ValueError:
        return
    np.testing.assert_array_equal(after.supported_source_indices, fresh.supported_source_indices)
    np.testing.assert_array_equal(after.target_indices, fresh.target_indices)
    np.testing.assert_allclose(after.apply(after.pack_source(changed.tensors[0])),
                               fresh.apply(fresh.pack_source(changed.tensors[0])))


def local_problem(s, hamiltonian, site, prune):
    frontier = ReducedFrontier.from_state(s, prune=prune)
    embedding = frontier.site_embedding(s, site)
    return _native_local_problem(s, hamiltonian, site, frontier.to_mps(s, compact=True),
                                 embedding, True, 0, structured=True)


def dense_frame(s, embedding, site):
    trial = s.copy()
    columns = []
    for vector in np.eye(embedding.source_size):
        trial.tensors[site] = embedding.unpack_source(vector)
        columns.append(trial.state_vector())
    return np.column_stack(columns)


@pytest.mark.parametrize('site', [0, 1, 2])
def test_norm_hopping_action_and_optimized_energy_match_unpruned_and_dense(site):
    s = state()
    h = np.array([[.1, -1., .25], [-1., -.2, -.8], [.25, -.8, .3]])
    g = np.zeros((3,) * 4)
    for i in range(3):
        g[i, i, i, i] = 4.
    electronic = ElectronicProblem(h, g, (2, 1))
    ham = electronic.su2_mpo()
    dense_h = determinant_hamiltonian(electronic)
    old = local_problem(s, ham, site, False)
    new = local_problem(s, ham, site, True)
    # Native local forms sum over all target magnetic components.
    frame = np.sqrt(s.symmetry.sector.irrep.dim) * dense_frame(s, new.embedding, site)
    support = new.embedding.supported_source_indices
    rng = np.random.default_rng(53)
    x = rng.normal(size=new.local_dimension) + 1j * rng.normal(size=new.local_dimension)
    raw = np.zeros(old.local_dimension, dtype=complex)
    raw[support] = x
    for name, reference in [('apply_metric', frame.conj().T @ frame),
                            ('apply_hamiltonian', frame.conj().T @ dense_h @ frame)]:
        value = getattr(new, name)(x)
        np.testing.assert_allclose(value, getattr(old, name)(raw)[support], atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(value, reference @ x, atol=2e-11, rtol=2e-11)
    options = LETTADMROptions(metric_tolerance=1e-12, eigensolver_tolerance=1e-10,
        eigensolver_max_iterations=300, matrix_free=True, dense_solver_threshold=0)
    energies = []
    for problem in (old, new):
        initial = problem.embedding.pack_source(s.tensors[site])
        energy, vector, *_ = _solve_local_problem(problem, options, initial_vector=initial)
        trial = s.copy()
        trial.tensors[site] = problem.embedding.unpack_source(vector)
        wavefunction = trial.state_vector()
        checked = np.vdot(wavefunction, dense_h @ wavefunction) / np.vdot(wavefunction, wavefunction)
        np.testing.assert_allclose(energy, checked, atol=2e-10)
        energies.append(checked)
    np.testing.assert_allclose(energies[0], energies[1], atol=2e-9)


def test_singlet_left_right_cycle_preserves_updates_and_full_source_shapes():
    h = np.zeros((4, 4))
    for i, j in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        h[i, j] = h[j, i] = -1.
    g = np.zeros((4,) * 4)
    for i in range(4):
        g[i, i, i, i] = 4.
    electronic = ElectronicProblem(h, g, (2, 2))
    s = ReducedLatticeLETTA.random((2, 2), symmetry=electronic.symmetry('su2'),
        seed=2031, real=False, normalize=False)
    ham = electronic.su2_mpo()
    dense_h = determinant_hamiltonian(electronic)
    shapes = [{key: value.shape for key, value in blocks.items()} for blocks in s.tensors]
    options = LETTADMROptions(metric_tolerance=1e-12, eigensolver_tolerance=1e-10,
        eigensolver_max_iterations=300, matrix_free=True, dense_solver_threshold=0)
    for site in (*range(4), *range(3, -1, -1)):
        energies = []
        for prune in (False, True):
            problem = local_problem(s, ham, site, prune)
            energy, vector, *_ = _solve_local_problem(problem, options,
                initial_vector=problem.embedding.pack_source(s.tensors[site]))
            trial = s.copy()
            trial.tensors[site] = problem.embedding.unpack_source(vector)
            assert {key: value.shape for key, value in trial.tensors[site].items()} == shapes[site]
            wavefunction = trial.state_vector()
            checked = np.vdot(wavefunction, dense_h @ wavefunction) / np.vdot(wavefunction, wavefunction)
            np.testing.assert_allclose(energy, checked, atol=3e-10)
            energies.append(checked)
        np.testing.assert_allclose(energies[0], energies[1], atol=3e-9)
        s = trial


def test_energy_prefix_cache_invalidates_distant_structural_support_change():
    from pyqed._letta_one_site_opt.frontier_contraction import EnergyBoundaryCache, frontier_energy
    s = state()
    electronic = ElectronicProblem(np.eye(3), np.zeros((3,) * 4), (2, 1))
    mpo = electronic.su2_mpo().native_mpo(s.physical_basis)
    cache = EnergyBoundaryCache(16 * 1024**2)
    frontier_energy(s, mpo, cache=cache)
    assert cache.entries and cache.nbytes > 0
    before = cache.signatures(s, mpo)
    changed = s.copy()
    changed.tensors[-1].pop(next(iter(changed.tensors[-1])))
    assert changed.bond_sectors == s.bond_sectors
    assert changed.neighborhoods == s.neighborhoods
    after = cache.signatures(changed, mpo)
    # Prefix values are identical, but the supported global paths are not.
    assert after[:-1] == before[:-1]
    assert not cache.entries and cache.nbytes == 0
    assert cache.nearest(after) is None
    np.testing.assert_allclose(frontier_energy(changed, mpo, cache=cache),
                               frontier_energy(changed, mpo), atol=1e-13)


def test_canonical_boundary_cache_uses_contraction_inputs_not_graph_name():
    from pyqed._letta_one_site_opt.frontier_contraction import CanonicalBoundaryCache, canonical_environment
    s = state()
    frontier = ReducedFrontier.from_state(s, prune=True)
    sites = frontier.to_mps(s, compact=True)
    fingerprint = frontier.support(s).fingerprint
    assert all(tensor.metadata['letta_frontier_support'] == fingerprint for tensor in sites)
    electronic = ElectronicProblem(np.eye(3), np.zeros((3,) * 4), (2, 1))
    mpo = electronic.su2_mpo().native_mpo(s.physical_basis)
    cache = CanonicalBoundaryCache(16 * 1024**2)
    canonical_environment(sites, 1, mpo, cache=cache, structured=True)
    assert cache.entries
    before = cache.signatures(sites, mpo)
    changed = s.copy()
    changed.tensors[-1].pop(next(iter(changed.tensors[-1])))
    new_fingerprint = ReducedFrontier.from_state(changed, prune=True).support(changed).fingerprint
    assert new_fingerprint != fingerprint
    # Metadata alone cannot change QR or the contraction. Actual pruned MPO
    # kernels are included by canonical_environment's boundary signatures.
    from pyqed._letta_one_site_opt.frontier_contraction import shallow_site
    changed_sites = [shallow_site(tensor) for tensor in sites]
    for tensor in changed_sites:
        tensor.metadata = dict(tensor.metadata, letta_frontier_support=new_fingerprint)
    after = cache.signatures(changed_sites, mpo)
    assert after==before
    _,_,_,cached=canonical_environment(changed_sites,1,mpo,cache=cache,structured=True)
    _,_,_,fresh=canonical_environment(changed_sites,1,mpo,structured=True)
    for side in ('left','right'):
        index=1 if side=='left' else 2
        for key,value in getattr(fresh,side)[index].items():
            np.testing.assert_allclose(getattr(cached,side)[index][key],value,atol=1e-13)


@pytest.mark.parametrize('equilibrated', [False, True])
def test_compact_source_local_projector_preserves_metric_support(equilibrated):
    from pyqed._letta_one_site_opt.reduced_norm import ReducedNormChain
    s = state(copies=3)
    frontier = ReducedFrontier.from_state(s, prune=True)
    chain = ReducedNormChain.build(frontier.to_mps(s, compact=True))
    rng = np.random.default_rng(2037)
    for i in range(s.nsites):
        embedding = frontier.site_embedding(s, i)
        projector = chain.local_projector(i, embedding, 1e-12, equilibrated=equilibrated)
        assert projector is not None
        def action(v):
            blocks = embedding.unpack_target(embedding.apply(v))
            return embedding.adjoint(embedding.pack_target(chain.local_action(i, blocks)))
        matrix = np.column_stack([action(x) for x in np.eye(embedding.source_size)])
        if equilibrated:
            roots = np.sqrt(np.maximum(np.diag(matrix).real, 0.))
            inverse = np.divide(1., roots, out=np.zeros_like(roots), where=roots > 0.)
            matrix = inverse[:, None] * matrix * inverse[None, :]
        x = rng.normal(size=embedding.source_size) + 1j * rng.normal(size=embedding.source_size)
        px = projector(x)
        np.testing.assert_allclose(matrix @ px, matrix @ x, atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(projector(px), px, atol=3e-12)
        assert np.linalg.norm(px) <= np.linalg.norm(x) + 1e-12
        with pytest.raises(ValueError, match='incompatible size'):
            projector(np.zeros(embedding.source_size + 1))


@pytest.mark.parametrize('prune', [False, True])
def test_source_vector_size_errors_are_explicit(prune):
    s = state()
    embedding = ReducedFrontier.from_state(s, prune=prune).site_embedding(s, 0)
    for count in (0, embedding.source_size - 1, embedding.source_size + 1):
        with pytest.raises(ValueError, match='incompatible size'):
            embedding.apply(np.zeros(count))
        with pytest.raises(ValueError, match='incompatible size'):
            embedding.unpack_source(np.zeros(count))
    with pytest.raises(ValueError, match='incompatible size'):
        embedding.adjoint(np.zeros(embedding.target_size + 1))


def test_current_support_graph_reused_above_shared_cache_budget(monkeypatch):
    from collections import OrderedDict
    from pyqed._letta_one_site_opt import frontier_support
    monkeypatch.setattr(frontier_support, '_SUPPORT_CACHE', OrderedDict())
    monkeypatch.setattr(frontier_support, '_SUPPORT_CACHE_BYTES', 0)
    monkeypatch.setattr(frontier_support, '_SUPPORT_CACHE_LIMIT', 1)
    s = state()
    frontier = ReducedFrontier.from_state(s, prune=True)
    graph = frontier.support(s)
    assert graph.cache_bytes > frontier_support._SUPPORT_CACHE_LIMIT
    assert not frontier_support._SUPPORT_CACHE
    for i in range(s.nsites):
        frontier.site_embedding(s, i)
        assert frontier.support(s) is graph
    changed_values = s.copy()
    for block in changed_values.tensors[0].values():
        block *= 1.25
    assert frontier.support(changed_values) is graph
    assert frontier_support.get_frontier_support(changed_values, frontier.cuts, previous=graph) is graph
    assert not frontier_support._SUPPORT_CACHE
    assert frontier_support._SUPPORT_CACHE_BYTES == 0


def test_global_support_change_clears_every_cached_embedding(monkeypatch):
    import gc
    import weakref
    from collections import OrderedDict
    from pyqed._letta_one_site_opt import frontier_support
    monkeypatch.setattr(frontier_support, '_SUPPORT_CACHE', OrderedDict())
    monkeypatch.setattr(frontier_support, '_SUPPORT_CACHE_BYTES', 0)
    monkeypatch.setattr(frontier_support, '_SUPPORT_CACHE_LIMIT', 1)
    s = state()
    frontier = ReducedFrontier.from_state(s, prune=True)
    old_graph = weakref.ref(frontier.support(s))
    for i in range(s.nsites):
        frontier.site_embedding(s, i)
    assert len(frontier._embedding_cache) == s.nsites
    assert frontier._embedding_cache_bytes > 0
    changed = s.copy()
    changed.tensors[-1].pop(next(iter(changed.tensors[-1])))
    graph = frontier.support(changed)
    assert graph is not old_graph()
    assert not frontier._embedding_cache
    assert frontier._embedding_cache_bytes == 0
    gc.collect()
    assert old_graph() is None
    frontier.site_embedding(changed, 0)
    assert tuple(frontier._embedding_cache) == (0,)
    assert frontier._embedding_cache[0][0] is graph
