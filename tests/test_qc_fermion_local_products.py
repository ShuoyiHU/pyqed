"""Exact local Jordan-Wigner products, including coincident orbital indices."""
import itertools
import numpy as np
import pytest
from pyqed.mps.nonabelian import models as m

@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('use_cache', [False, True])
def test_local_products_match_explicit_operator_composition(dtype, use_cache):
    cache = {} if use_cache else None
    leg = m.physical_leg_from_spatial_orbital()
    parity = m.spatial_parity(dtype=dtype)
    pairs = [(m.spatial_create_up(dtype=dtype), m.spatial_annihilate_up(dtype=dtype)),
             (m.spatial_create_down(dtype=dtype), m.spatial_annihilate_down(dtype=dtype))]
    for sites in itertools.product(range(4), repeat=4):
        for left, right in itertools.product(pairs, repeat=2):
            operators = left + right
            groups = {}
            for operator, site in zip(operators, sites):
                for j in range(site):
                    groups.setdefault(j, []).append(parity)
                groups.setdefault(site, []).append(operator)
            reference = []
            for j, local in sorted(groups.items()):
                dense = m.compose_site_operators(*local).as_dense()
                if not np.any(dense):
                    reference = []
                    break
                if not np.array_equal(dense, np.eye(4)):
                    reference.append((j, dense))
            result = m._canonical_spatial_jw_site_operators(
                operators, sites, phys_leg=leg, dtype=dtype, cutoff=0., cache=cache)
            assert [j for j, _ in result] == [j for j, _ in reference]
            for (_, actual), (_, expected) in zip(result, reference):
                np.testing.assert_array_equal(actual.as_dense(), expected)
