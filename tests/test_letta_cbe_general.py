"""Independent incidence and labelled-network checks for general CBE."""

import importlib
import itertools

import numpy as np
import pytest

from pyqed._letta_one_site_opt.contractions import _contract_operands


def _general():
    return importlib.import_module("pyqed._letta_one_site_opt.cbe_general")


CATEGORIES = tuple(
    "".join(subset)
    for size in range(1, 5)
    for subset in itertools.combinations("LABR", size)
)


@pytest.mark.parametrize("category", CATEGORIES)
def test_incidence_home_and_selection_role(category):
    # Two physical indices may share a category; the home is explicit.
    region_site = dict(zip("LABR", range(4)))
    neighborhoods = tuple(
        tuple(p for p in range(2) if region in category)
        for region in "LABR"
    )
    inventory = _general().PhysicalIndexInventory.from_dependencies(
        neighborhoods, homes=(region_site[category[-1]],) * 2,
        dimensions=(2, 3), left_site=1,
    )
    assert len(inventory.indices) == 2
    expected = ("shared" if "A" in category and "B" in category else
                "row" if "A" in category else
                "column" if "B" in category else "environment")
    for item in inventory.indices:
        assert item.regions == tuple(category)
        assert item.home_region == category[-1]
        assert item.role == expected
    assert inventory.categories[category] == (0, 1)


def test_home_is_not_inferred_from_last_incidence():
    inventory = _general().PhysicalIndexInventory.from_dependencies(
        ((0,), (), (), (0,)), homes=(0,), dimensions=(2,), left_site=1,
    )
    assert inventory.indices[0].regions == ("L", "R")
    assert inventory.indices[0].home_region == "L"
    assert inventory.indices[0].role == "environment"


def _network(categories, seed=55):
    """Unsplit bra-environment/H/ket network, without production routing.

    Each category adds an independent physical index and a random complex
    local Hamiltonian factor at its declared home. The old ket has bond 2;
    outer virtual sizes 2 and 3 deliberately differ. No active bra factors
    are included: their arguments are precisely the requested output.
    """
    rng = np.random.default_rng(seed)
    categories = ("L", "A", "B", "R") + tuple(categories)
    dims = {0: 2, 1: 2, 2: 3, 3: 2, 4: 2, 5: 3}
    ket = tuple(100 + p for p in range(len(categories)))
    bra = tuple(200 + p for p in range(len(categories)))
    for p in range(len(categories)):
        dims[ket[p]] = dims[bra[p]] = 2 + (p == 1)

    def random(labels):
        shape = tuple(dims[label] for label in labels)
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    args = {r: tuple(p for p, cat in enumerate(categories) if r in cat)
            for r in "LABR"}
    lk = (0,) + tuple(ket[p] for p in args["L"])
    rk = (2,) + tuple(ket[p] for p in args["R"])
    lb = (3,) + tuple(bra[p] for p in args["L"])
    rb = (5,) + tuple(bra[p] for p in args["R"])
    a = (0,) + tuple(ket[p] for p in args["A"]) + (1,)
    b = (1,) + tuple(ket[p] for p in args["B"]) + (2,)
    el, er = random(lk), random(rk)
    left = [(el.conj(), lb), (el, lk), (random(a), a)]
    right = [(random(b), b), (er.conj(), rb), (er, rk)]
    for p, cat in enumerate(categories):
        labels = (bra[p], ket[p])
        (left if cat[-1] in "LA" else right).append((random(labels), labels))
    shared = tuple(bra[p] for p in args["A"] if p in args["B"])
    rows = (3,) + tuple(bra[p] for p in args["A"] if p not in args["B"])
    columns = tuple(bra[p] for p in args["B"] if p not in args["A"]) + (5,)
    return left, right, rows, columns, shared, dims


@pytest.mark.parametrize("categories", [(c,) for c in CATEGORIES] + [
    ("LR", "AR", "LBR"), ("LABR", "ABR", "LR"),
    ("LB", "LAR", "AB"), ("AR", "AR", "BR"),
])
def test_routed_halves_equal_unsplit_complex_response(categories):
    left, right, rows, columns, shared, dims = _network(categories)
    route = _general().ResponseRouting.from_operands(
        left, right, rows=rows, columns=columns, shared=shared, dimensions=dims,
    )
    # Independent reference: original factors and no inferred frontier or lift.
    operands, labels = zip(*(left + right))
    direct = _contract_operands(operands, labels, rows + shared + columns)
    for block in np.ndindex(*(dims[s] for s in shared)):
        halves = route.half_matrices(block)
        section = (slice(None),) * len(rows) + block + (slice(None),) * len(columns)
        reference = direct[section].reshape(halves.left.shape[0], halves.right.shape[1])
        np.testing.assert_allclose(halves.left @ halves.right, reference,
                                   rtol=2e-12, atol=2e-9)
        assert not set(rows + columns + shared) & set(halves.connectors)


def test_missing_nontrivial_output_is_rejected_not_broadcast():
    with pytest.raises(ValueError, match="missing.*output"):
        _general().ResponseRouting.from_operands(
            [(np.ones(2), (0,))], [(np.ones(2), (0,))],
            rows=(1,), columns=(), shared=(), dimensions={0: 2, 1: 2},
        )


def _lattice_problem(shape, left_site, seed=910):
    from pyqed._letta_one_site_opt import LatticeLETTA
    from pyqed._letta_one_site_opt.operators import LatticeMPO
    from pyqed._letta_two_site_opt import LETTAPairLayout, LETTAPairEnvironmentCache
    rng = np.random.default_rng(seed)
    state = LatticeLETTA.random(shape, physical_dim=2, bond_dim=2, seed=seed)
    factors = []
    for k in range(state.nsites):
        w = (1 if k == 0 else 2, 1 if k == state.nsites - 1 else 2, 2, 2)
        factors.append(rng.normal(size=w) + 1j * rng.normal(size=w))
    mpo = LatticeMPO(factors, lattice_shape=shape)
    layout = LETTAPairLayout.from_state(state, left_site)
    cache = LETTAPairEnvironmentCache(state, mpo)
    hl = cache.build_left_environments()[left_site]
    hr = cache.build_right_environments()[left_site + 2]
    return state, layout, cache, hl, hr


@pytest.mark.parametrize("shape,site", [((2, 3), 0), ((3, 3), 1),
                                       ((3, 3), 3), ((2, 2, 2), 2)])
def test_real_lattice_response_routing_matches_pair_action(shape, site):
    state, layout, cache, hl, hr = _lattice_problem(shape, site)
    a, b = state.tensors[site:site + 2]
    route = _general().ResponseRouting.from_cache(cache, hl, hr, layout, a, b)
    inventory = _general().PhysicalIndexInventory.from_state(state, site)
    assert tuple(p.site for p in inventory.indices if p.role == "shared") == tuple(sorted(layout.shared))
    target = cache.effective_pair_action(hl, hr, layout, layout.merge(a, b).ravel()).reshape(layout.merged_shape)
    for block in np.ndindex(*((2,) * len(layout.shared))):
        halves = route.half_matrices(block)
        section = (slice(None),) * (1 + len(layout.left_only)) + block + (slice(None),) * (1 + len(layout.right_only))
        np.testing.assert_allclose(halves.left @ halves.right,
                                   target[section].reshape(halves.left.shape[0], -1),
                                   rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_conditional_candidates_equal_dense_double_projection(direction):
    state, layout, cache, hl, hr = _lattice_problem((2, 3), 1)
    a, b = state.tensors[1:3]
    proposal = _general().conditional_candidates(
        cache, hl, hr, layout, a, b, direction=direction, width=1,
    )
    target = cache.effective_pair_action(hl, hr, layout, layout.merge(a, b).ravel()).reshape(layout.merged_shape)
    for block in np.ndindex(*((2,) * len(layout.shared))):
        asec, bsec = [slice(None)] * a.ndim, [slice(None)] * b.ndim
        for p, q in zip(layout.shared, block):
            asec[1 + layout.left_neighborhood.index(p)] = q
            bsec[1 + layout.right_neighborhood.index(p)] = q
        am = a[tuple(asec)].reshape(-1, a.shape[-1])
        bm = b[tuple(bsec)].reshape(b.shape[0], -1)
        sec = (slice(None),) * (1 + len(layout.left_only)) + block + (slice(None),) * (1 + len(layout.right_only))
        response = target[sec].reshape(am.shape[0], bm.shape[1])
        missing = (np.eye(am.shape[0]) - am @ np.linalg.pinv(am)) @ response @ (np.eye(bm.shape[1]) - np.linalg.pinv(bm) @ bm)
        u, _, vh = np.linalg.svd(missing, full_matrices=False)
        if direction == "rl":
            got = proposal.tensor[tuple(asec)].reshape(am.shape[0], -1)
            expected = u[:, :1]
            np.testing.assert_allclose(got @ got.conj().T, expected @ expected.conj().T, atol=2e-10)
        else:
            got = proposal.tensor[tuple(bsec)].reshape(-1, bm.shape[1])
            expected = vh[:1]
            np.testing.assert_allclose(got.conj().T @ got, expected.conj().T @ expected, atol=2e-10)


@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_conditional_candidates_invariant_to_environment_only_gauge(direction):
    from pyqed._letta_two_site_opt import LETTAPairEnvironmentCache
    state, layout, cache, hl, hr = _lattice_problem((3, 3), 1)
    transformed = state.copy()
    for site, factors in ((0, [1., 7.]), (3, [1., 1. / 7.])):
        shape = [1] * transformed.tensors[site].ndim
        shape[1 + transformed.site_neighborhood(site).index(3)] = 2
        transformed.tensors[site] *= np.array(factors).reshape(shape)
    other = LETTAPairEnvironmentCache(transformed, cache.mpo)
    proposals = []
    for s, h, l, r in ((state, cache, hl, hr),
                       (transformed, other, other.build_left_environments()[1],
                        other.build_right_environments()[3])):
        proposals.append(_general().conditional_candidates(
            h, l, r, layout, *s.tensors[1:3], direction=direction, width=1,
        ).tensor)
    for block in np.ndindex(*((2,) * len(layout.shared))):
        tensors = state.tensors[1] if direction == "rl" else state.tensors[2]
        neighborhood = layout.left_neighborhood if direction == "rl" else layout.right_neighborhood
        section = [slice(None)] * tensors.ndim
        for p, q in zip(layout.shared, block):
            section[1 + neighborhood.index(p)] = q
        matrices = [p[tuple(section)] for p in proposals]
        if direction == "rl":
            matrices = [p.reshape(-1, p.shape[-1]) for p in matrices]
            projectors = [p @ np.linalg.pinv(p) for p in matrices]
        else:
            matrices = [p.reshape(p.shape[0], -1) for p in matrices]
            projectors = [np.linalg.pinv(p) @ p for p in matrices]
        np.testing.assert_allclose(*projectors, atol=2e-10)


def _frame_oracle(layout, frame):
    # Deliberately use the dense pair embedding only in the oracle.
    basis = np.eye(np.prod(frame.shape), dtype=complex)
    return np.column_stack([
        (layout.merge(v.reshape(frame.shape), frame.fixed) if frame.side == "A"
         else layout.merge(frame.fixed, v.reshape(frame.shape))).ravel()
        for v in basis
    ])


@pytest.mark.parametrize("shape,site", [((2, 3), 1), ((3, 3), 1), ((2, 2, 2), 2)])
def test_cross_frame_gram_actions_match_dense_pair_oracle(shape, site):
    from pyqed._letta_two_site_opt import IdentityPairEnvironmentCache
    state, layout, h, hl, hr = _lattice_problem(shape, site)
    a, b = state.tensors[site:site + 2]
    n = IdentityPairEnvironmentCache(state)
    nl, nr = n.build_left_environments()[site], n.build_right_environments()[site + 2]
    network = _general().FixedPairFrames(n, nl, nr, layout)
    rng = np.random.default_rng(119)
    ap = rng.normal(size=a.shape[:-1] + (1,)) + 1j * rng.normal(size=a.shape[:-1] + (1,))
    bp = rng.normal(size=(1,) + b.shape[1:]) + 1j * rng.normal(size=(1,) + b.shape[1:])
    frames = [network.frame("A", b), network.frame("B", a),
              network.frame("A", bp), network.frame("B", ap)]
    pair_metric = n.effective_pair_metric(nl, nr, layout).to_dense()
    embeddings = [_frame_oracle(layout, frame) for frame in frames]
    for i, bra in enumerate(frames):
        for j, ket in enumerate(frames):
            vector = rng.normal(size=np.prod(ket.shape)) + 1j * rng.normal(size=np.prod(ket.shape))
            expected = embeddings[i].conj().T @ pair_metric @ embeddings[j] @ vector
            got = network.overlap(bra, ket, vector)
            np.testing.assert_allclose(got, expected, rtol=3e-10, atol=2e-9)
        np.testing.assert_allclose(network.metric(bra).to_dense(),
                                   embeddings[i].conj().T @ pair_metric @ embeddings[i],
                                   rtol=3e-10, atol=2e-9)


@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_restricted_physical_objective_matches_full_tangent_oracle(direction):
    from pyqed._letta_two_site_opt import IdentityPairEnvironmentCache
    state, layout, h, hl, hr = _lattice_problem((2, 3), 1)
    a, b = state.tensors[1:3]
    n = IdentityPairEnvironmentCache(state)
    nl, nr = n.build_left_environments()[1], n.build_right_environments()[3]
    network = _general().FixedPairFrames(n, nl, nr, layout)
    proposal = _general().conditional_candidates(h, hl, hr, layout, a, b,
                                                direction=direction, width=2)
    result = _general().restricted_physical_problem(
        h, hl, hr, network, a, b, proposal.tensor, direction=direction,
        energy=0.3, tolerance=1e-11, max_iterations=1000,
    )
    fa, fb = network.frame("A", b), network.frame("B", a)
    fpr = network.frame("B", proposal.tensor) if direction == "rl" else network.frame("A", proposal.tensor)
    ja, jb, jp = [_frame_oracle(layout, f) for f in (fa, fb, fpr)]
    metric = n.effective_pair_metric(nl, nr, layout).to_dense()
    theta = layout.merge(a, b).ravel()
    response = h.effective_pair_action(hl, hr, layout, theta) - .3 * metric @ theta
    tangent = np.column_stack([ja, jb])
    gram = tangent.conj().T @ metric @ tangent
    expected_rhs = jp.conj().T @ response - jp.conj().T @ metric @ tangent @ np.linalg.pinv(gram, rcond=1e-11, hermitian=True) @ (tangent.conj().T @ response)
    old = jb if direction == "rl" else ja
    cross = old.conj().T @ metric @ jp
    expected_metric = jp.conj().T @ metric @ jp - cross.conj().T @ np.linalg.pinv(old.conj().T @ metric @ old, rcond=1e-11, hermitian=True) @ cross
    np.testing.assert_allclose(result.rhs, expected_rhs, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(result.metric.to_dense(), expected_metric, rtol=2e-8, atol=2e-9)
    assert result.tangent_relative_residual < 1e-8
    assert len(result.tangent_block_shapes) == 2 ** len(layout.shared)
    assert all(rows > 0 and columns > 0 for rows, columns in result.tangent_block_shapes)
    assert max(rows for rows, _ in result.tangent_block_shapes) < ja.shape[1]
    assert max(columns for _, columns in result.tangent_block_shapes) < jb.shape[1]


@pytest.mark.parametrize("singular", [False, True])
def test_separable_metric_fit_matches_supported_whitened_svd(singular):
    from pyqed._letta_one_site_opt.contractions import BlockDiagonalMetric
    rng = np.random.default_rng(633)
    ul, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    ur, _ = np.linalg.qr(rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5)))
    vl, vr = np.arange(1., 5.), np.arange(1., 6.)
    if singular:
        vl[0], vr[0] = 0., 0.
    gl, gr = (ul * vl) @ ul.conj().T, (ur * vr) @ ur.conj().T
    dense = np.kron(gl, gr)
    metric = BlockDiagonalMetric(20, [dense], [np.arange(20)])
    target = rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))
    rhs = dense @ target.ravel()
    fit = _general().fit_low_rank_quadratic(rhs, metric, (4, 5), rank=1)
    white = (np.sqrt(vl)[:, None] * (ul.conj().T @ target @ ur.conj()) * np.sqrt(vr)[None, :])
    singular_values = np.linalg.svd(white, compute_uv=False)
    assert fit.metric_kind == "separable"
    assert fit.iterations == 0
    np.testing.assert_allclose(fit.captured_weight, singular_values[0] ** 2,
                               rtol=1e-10, atol=1e-10)


def test_nonseparable_metric_fit_improves_physical_objective():
    from pyqed._letta_one_site_opt.contractions import BlockDiagonalMetric
    rng = np.random.default_rng(701)
    q = rng.normal(size=(20, 20)) + 1j * rng.normal(size=(20, 20))
    dense = q.conj().T @ q + np.eye(20)
    metric = BlockDiagonalMetric(20, [dense], [np.arange(20)])
    target = rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))
    rhs = dense @ target.ravel()
    u, s, vh = np.linalg.svd(target, full_matrices=False)
    initial = (u[:, :1] * s[:1]) @ vh[:1]
    initial_error = (target - initial).ravel()
    initial_loss = np.real(np.vdot(initial_error, dense @ initial_error))
    fit = _general().fit_low_rank_quadratic(rhs, metric, target.shape, rank=1,
                                          max_iterations=20)
    assert fit.metric_kind == "general"
    assert fit.iterations > 0
    assert fit.loss < initial_loss * .95
    assert np.linalg.matrix_rank(fit.left @ fit.right, tol=1e-9) == 1
    error = (target - fit.left @ fit.right).ravel()
    np.testing.assert_allclose(fit.loss, np.real(np.vdot(error, dense @ error)), rtol=1e-10)


def test_zero_metric_fit_does_not_promote_null_modes():
    from pyqed._letta_one_site_opt.contractions import BlockDiagonalMetric
    metric = BlockDiagonalMetric(6, [np.zeros((6, 6))], [np.arange(6)])
    fit = _general().fit_low_rank_quadratic(np.zeros(6), metric, (2, 3), rank=1)
    assert fit.captured_weight == 0.
    assert np.count_nonzero(fit.left) == 0
    assert np.count_nonzero(fit.right) == 0


def test_separable_physical_fit_keeps_small_independent_weights():
    from pyqed._letta_one_site_opt.contractions import DiagonalMetric
    scales = np.array([1e-10, 1., 1e10])
    metric = DiagonalMetric(scales**2)
    target = (1. / scales).reshape(3, 1)
    fit = _general().fit_low_rank_quadratic(metric @ target.ravel(), metric,
                                           target.shape, rank=1)
    np.testing.assert_allclose(scales * (fit.left @ fit.right).ravel(), 1., atol=1e-12)
    np.testing.assert_allclose(fit.available_weight, 3., atol=1e-12)
    np.testing.assert_allclose(fit.captured_weight, 3., atol=1e-12)


def test_candidate_projection_distinguishes_small_weights_from_cancellation():
    # First candidate repeats the small old direction; second is physically
    # independent with equally small coordinates. Only the second may survive.
    nn = np.diag([1e-20, 1e-20])
    oo = np.diag([1e-20, 1.])
    on = np.diag([1e-20, 0.])
    residual = _general()._supported_schur_complement(nn, oo, on, 1e-12)
    np.testing.assert_allclose(residual / 1e-20, np.diag([0., 1.]), atol=1e-12)


@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_production_selector_uses_general_physical_geometry(direction):
    from pyqed._letta_two_site_opt import IdentityPairEnvironmentCache, LETTAPairEnvironmentCache
    from pyqed._letta_one_site_opt.cbe import streamed_shrewd_cbe_selection
    state, layout, h, hl, hr = _lattice_problem((3, 3), 1)
    transformed = state.copy()
    for site, values in ((0, [1., 7.]), (3, [1., 1. / 7.])):
        dims = [1] * transformed.tensors[site].ndim
        dims[1 + transformed.site_neighborhood(site).index(3)] = 2
        transformed.tensors[site] *= np.array(values).reshape(dims)
    outputs = []
    for s in (state, transformed):
        h = LETTAPairEnvironmentCache(s, h.mpo)
        n = IdentityPairEnvironmentCache(s)
        result = streamed_shrewd_cbe_selection(
            h, h.build_left_environments()[1], h.build_right_environments()[3],
            layout, *s.tensors[1:3], expansion_dimension=1,
            preselection_dimension=2, direction=direction, metric_cache=n,
            metric_left=n.build_left_environments()[1],
            metric_right=n.build_right_environments()[3], energy=.3,
        )
        assert result.pair_action_count == result.pair_metric_count == result.merged_pair_count == 0
        assert len(result.sector_ranks) == 2 ** len(layout.shared)
        assert result.overlap_applications > 0
        assert result.tangent_block_shapes
        assert set(result.selection_timings) == {
            "routing", "preselection", "old_metrics", "response",
            "tangent_projection", "schur_metric", "restricted_fit",
        }
        assert all(np.isfinite(t) and t >= 0. for t in result.selection_timings.values())
        outputs.append(result.left_direction if direction == "rl" else result.right_direction)
    neighborhood = layout.left_neighborhood if direction == "rl" else layout.right_neighborhood
    for block in np.ndindex(*((2,) * len(layout.shared))):
        section = [slice(None)] * outputs[0].ndim
        for p, q in zip(layout.shared, block):
            section[1 + neighborhood.index(p)] = q
        matrices = [value[tuple(section)] for value in outputs]
        if direction == "rl":
            matrices = [m.reshape(-1, m.shape[-1]) for m in matrices]
            projections = [m @ np.linalg.pinv(m) for m in matrices]
        else:
            matrices = [m.reshape(m.shape[0], -1) for m in matrices]
            projections = [np.linalg.pinv(m) @ m for m in matrices]
        np.testing.assert_allclose(*projections, atol=2e-7)


def _incidence_state(categories, *, zero_support=False):
    """Six-site fixture with explicit ties, using real LETTA/cache machinery."""
    from pyqed._letta_one_site_opt import LatticeLETTA
    neighborhoods = [set((k,)) for k in range(6)]
    # Extra indices have real distinct homes. The mixed examples use two R
    # homes and a B home, so repeated categories are not collapsed to one axis.
    used_homes = set()
    for category in categories:
        choices = {"L": (0, 1), "A": (2,), "B": (3,), "R": (5, 4)}[category[-1]]
        home = next(k for k in choices if k not in used_homes)
        used_homes.add(home)
        for region in category:
            dependent = {"L": 0, "A": 2, "B": 3, "R": home}[region]
            neighborhoods[dependent].add(home)
    neighborhoods = tuple((k,) + tuple(sorted(ns - {k})) for k, ns in enumerate(neighborhoods))

    rng = np.random.default_rng(941)
    tensors = []
    for k, ns in enumerate(neighborhoods):
        shape = (1 if k == 0 else 2,) + (2,) * len(ns) + (1 if k == 5 else 2,)
        tensors.append(rng.normal(size=shape) + 1j * rng.normal(size=shape))
    if zero_support:
        tensors[5][:, 1, :] = 0.
    return LatticeLETTA((1, 6), 2, tensors, neighborhoods=neighborhoods)


def _physical_frame_by_amplitudes(state, frame):
    # No pair merge, environment contraction or production frame algebra.
    # Enumerate physical configurations and multiply ordinary virtual matrices.
    result = np.zeros((state.hilbert_dim, np.prod(frame.shape)), dtype=complex)
    for column in range(result.shape[1]):
        variable = np.zeros(frame.shape, dtype=complex)
        variable.flat[column] = 1.
        tensors = list(state.tensors)
        tensors[2:4] = [variable, frame.fixed] if frame.side == "A" else [frame.fixed, variable]
        for row, physical in enumerate(np.ndindex(*((2,) * state.nsites))):
            product = np.ones((1, 1), dtype=complex)
            for k, tensor in enumerate(tensors):
                section = (slice(None),) + tuple(physical[p] for p in state.site_neighborhood(k)) + (slice(None),)
                product = product @ tensor[section]
            result[row, column] = product[0, 0]
    return result


@pytest.mark.parametrize("categories", [(c,) for c in CATEGORIES] + [
    ("LR", "AR", "LB"), ("LABR", "LBR", "AB"), ("AR", "AR"),
])
@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_category_physical_objective_matches_amplitude_oracle(categories, direction):
    state = _incidence_state(categories, zero_support=True)
    _assert_physical_objective_by_amplitudes(state, direction)


@pytest.mark.parametrize("shape", [(2, 3), (2, 2, 2)])
@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_backward_dependencies_physical_objective_matches_amplitudes(shape, direction):
    from pyqed._letta_one_site_opt import LatticeLETTA
    coordinates = tuple(reversed(tuple(np.ndindex(*shape))))
    state = LatticeLETTA.random(shape, physical_dim=2, bond_dim=2, seed=943,
                                real=False, coordinates=coordinates)
    inventory = _general().PhysicalIndexInventory.from_state(state, 2)
    assert any(item.home < max(item.dependent_sites) for item in inventory.indices)
    _assert_physical_objective_by_amplitudes(state, direction)


def _assert_physical_objective_by_amplitudes(state, direction):
    from pyqed._letta_two_site_opt import IdentityPairEnvironmentCache, LETTAPairEnvironmentCache, LETTAPairLayout
    from pyqed._letta_one_site_opt.operators import LatticeMPO
    layout = LETTAPairLayout.from_state(state, 2)
    a, b = state.tensors[2:4]
    rng = np.random.default_rng(942)
    local = []
    for _ in range(state.nsites):
        matrix = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        local.append((matrix + matrix.conj().T).reshape(1, 1, 2, 2))
    mpo = LatticeMPO(local, lattice_shape=state.lattice_shape)
    n, h = IdentityPairEnvironmentCache(state), LETTAPairEnvironmentCache(state, mpo)
    nl, nr = n.build_left_environments()[2], n.build_right_environments()[4]
    network = _general().FixedPairFrames(n, nl, nr, layout)
    shape = a.shape[:-1] + (1,) if direction == "rl" else (1,) + b.shape[1:]
    pre = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    result = _general().restricted_physical_problem(
        h, h.build_left_environments()[2], h.build_right_environments()[4],
        network, a, b, pre, direction=direction, energy=.17, tolerance=1e-11,
    )
    frames = [network.frame("A", b), network.frame("B", a),
              network.frame("B" if direction == "rl" else "A", pre)]
    fa, fb, fp = [_physical_frame_by_amplitudes(state, frame) for frame in frames]
    residual = (mpo.to_dense() - .17 * np.eye(state.hilbert_dim)) @ state.state_vector()
    tangent = np.column_stack((fa, fb))
    u, singular, _ = np.linalg.svd(tangent, full_matrices=False)
    supported = u[:, singular > 1e-5 * singular[0]]
    missing = residual - supported @ (supported.conj().T @ residual)
    old = fb if direction == "rl" else fa
    u, singular, _ = np.linalg.svd(old, full_matrices=False)
    supported = u[:, singular > 1e-5 * singular[0]]
    new_frame = fp - supported @ (supported.conj().T @ fp)
    np.testing.assert_allclose(result.rhs, fp.conj().T @ missing, atol=3e-8, rtol=3e-7)
    np.testing.assert_allclose(result.metric.to_dense(), new_frame.conj().T @ new_frame, atol=2e-9, rtol=2e-8)
