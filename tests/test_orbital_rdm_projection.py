from types import SimpleNamespace

import numpy as np

from pyqed.qchem.dmrg.dmrg import DMRG
from pyqed.qchem.mcscf.cocas import _make_orbital_rdm12


def _eightfold_average(tensor):
    permutations = (
        (0, 1, 2, 3),
        (1, 0, 2, 3),
        (0, 1, 3, 2),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (2, 3, 1, 0),
        (3, 2, 0, 1),
        (3, 2, 1, 0),
    )
    return sum(np.transpose(tensor, p) for p in permutations) / 8.0


def test_fully_reduced_orbital_rdm_projects_only_active_two_rdm():
    rng = np.random.default_rng(17)
    ncore = 2
    ncas = 4
    norb = ncore + ncas
    dm1 = rng.normal(size=(norb, norb))
    dm2 = rng.normal(size=(norb,) * 4)
    source_dm2 = dm2.copy()
    solver = SimpleNamespace(
        site="spatial",
        spatial_site_basis="fully_reduced",
        ncore=ncore,
        ncas=ncas,
        make_rdm12=lambda *args, **kwargs: (dm1, dm2),
    )

    actual1, actual2 = DMRG.make_orbital_rdm12(
        solver,
        0,
        with_core=True,
    )

    expected_active = _eightfold_average(
        source_dm2[ncore:, ncore:, ncore:, ncore:]
    )
    np.testing.assert_array_equal(actual1, dm1)
    np.testing.assert_allclose(
        actual2[ncore:, ncore:, ncore:, ncore:],
        expected_active,
    )
    outside_active = np.ones(source_dm2.shape, dtype=bool)
    outside_active[ncore:, ncore:, ncore:, ncore:] = False
    np.testing.assert_array_equal(
        actual2[outside_active],
        source_dm2[outside_active],
    )
    assert not np.shares_memory(actual2, source_dm2)


def test_cocas_prefers_dedicated_orbital_rdm_and_has_generic_fallback():
    calls = []
    dedicated = SimpleNamespace(
        make_orbital_rdm12=lambda state, with_core: calls.append(
            ("orbital", state, with_core)
        ) or (1, 2),
        make_rdm12=lambda state, with_core: calls.append(
            ("generic", state, with_core)
        ) or (3, 4),
    )

    assert _make_orbital_rdm12(dedicated, 7, with_core=True) == (1, 2)
    assert calls == [("orbital", 7, True)]

    fallback = SimpleNamespace(
        make_rdm12=lambda state, with_core: (state, with_core),
    )
    assert _make_orbital_rdm12(fallback, 9, with_core=False) == (9, False)
