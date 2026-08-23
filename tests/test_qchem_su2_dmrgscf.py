from contextlib import redirect_stdout
from io import StringIO
import pickle
from time import perf_counter
from unittest import TestCase

import numpy as np

from pyqed.qchem import CASCI, Molecule
from pyqed.qchem.dmrg.backends.nonabelian import _qchem_sweep_measure
from pyqed.qchem.dmrg.dmrgscf import DMRGSCF
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf.cocas import (
    _active_core_energy,
    _active_core_gradient,
    energy as _embedded_rdm_energy,
)
from pyqed.optimize import gradient as _embedded_rdm_gradient


def _h2_rhf():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(
        driver="builtin",
        eri="dense",
        aosym="s1",
        options={"eri_backend": "cpp"},
    )
    return RHF(mol).run()


def _h2_factor_rhf():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(
        driver="builtin",
        eri="factors",
        options={"eri_backend": "cpp"},
    )
    return RHF(mol).run()


def _h4_rhf():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(
        driver="builtin",
        eri="dense",
        aosym="s1",
        options={"eri_backend": "cpp"},
    )
    return RHF(mol).run()


def _lih_rhf():
    mol = Molecule(
        atom="Li 0 0 0; H 0 0 3.0",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(
        driver="builtin",
        eri="dense",
        aosym="s1",
        options={"eri_backend": "cpp"},
    )
    return RHF(mol).run()


def _lih_factor_rhf():
    mol = Molecule(
        atom="Li 0 0 0; H 0 0 3.0",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(
        driver="builtin",
        eri="factors",
        options={"eri_backend": "cpp"},
    )
    return RHF(mol).run()


def _eightfold_eri_average(tensor):
    axes = (
        (0, 1, 2, 3),
        (1, 0, 2, 3),
        (0, 1, 3, 2),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (2, 3, 1, 0),
        (3, 2, 0, 1),
        (3, 2, 1, 0),
    )
    return sum(np.transpose(tensor, permutation) for permutation in axes) / 8.0


def _embed_closed_shell_core_rdm12(active_dm1, active_dm2, ncore):
    ncas = active_dm1.shape[0]
    norb = ncore + ncas
    dm1 = np.zeros((norb, norb), dtype=active_dm1.dtype)
    dm2 = np.zeros((norb, norb, norb, norb), dtype=active_dm2.dtype)
    np.fill_diagonal(dm1[:ncore, :ncore], 2.0)
    dm1[ncore:, ncore:] = active_dm1
    if ncore:
        eye = np.eye(ncore)
        dm2[:ncore, :ncore, :ncore, :ncore] = (
            4 * np.einsum("ij,kl->ijkl", eye, eye)
            - 2 * np.einsum("ps,rq->pqrs", eye, eye)
        )
        for i in range(ncore):
            dm2[i, i, ncore:, ncore:] = 2 * active_dm1
            dm2[ncore:, ncore:, i, i] = 2 * active_dm1
            dm2[i, ncore:, i, ncore:] = -active_dm1
            dm2[ncore:, i, ncore:, i] = -active_dm1
    dm2[ncore:, ncore:, ncore:, ncore:] = active_dm2
    return dm1, dm2


def test_active_only_rdm_analytic_core_objective_matches_embedded_rdm():
    rng = np.random.default_rng(7)
    nmo, ncore, ncas, naux = 7, 2, 3, 5
    U, _ = np.linalg.qr(rng.normal(size=(nmo, ncore + ncas)))
    h1e = rng.normal(size=(nmo, nmo))
    h1e = 0.5 * (h1e + h1e.T)
    pair_factors = rng.normal(size=(naux, nmo, nmo))
    pair_factors = 0.5 * (pair_factors + pair_factors.transpose(0, 2, 1))
    eri = np.einsum("Ppq,Prs->pqrs", pair_factors, pair_factors)
    active_dm1 = rng.normal(size=(ncas, ncas))
    active_dm1 = 0.5 * (active_dm1 + active_dm1.T)
    active_dm2 = rng.normal(size=(ncas, ncas, ncas, ncas))
    embedded_dm1, embedded_dm2 = _embed_closed_shell_core_rdm12(
        active_dm1,
        active_dm2,
        ncore,
    )

    reference_energy = _embedded_rdm_energy(
        U,
        h1e,
        eri,
        embedded_dm1,
        embedded_dm2,
    )
    reference_gradient = _embedded_rdm_gradient(
        U,
        h1e,
        eri,
        embedded_dm1,
        embedded_dm2,
    )
    for integral_representation in (eri, pair_factors):
        np.testing.assert_allclose(
            _active_core_energy(
                U,
                h1e,
                integral_representation,
                active_dm1,
                active_dm2,
                ncore,
            ),
            reference_energy,
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            _active_core_gradient(
                U,
                h1e,
                integral_representation,
                active_dm1,
                active_dm2,
                ncore,
            ),
            reference_gradient,
            atol=1.0e-9,
        )


def test_su2_dmrgscf_active_only_core_mode_matches_embedded_reference():
    mf = _lih_rhf()
    common = dict(
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=4,
        symmetry="su2",
        orbital_rdm_algorithm="npdm",
        init_guess="hf",
        verbose=0,
    )
    run_options = dict(
        nstates=1,
        nsweeps=3,
        mixer_zero_block_noise_scale=0.0,
        require_conv=False,
        warm_start_dmrg=False,
    )
    embedded = DMRGSCF(
        mf,
        orbital_core_mode="embedded",
        **common,
    ).run(**run_options)
    analytic = DMRGSCF(
        mf,
        orbital_core_mode="analytic",
        **common,
    ).run(**run_options)

    np.testing.assert_allclose(analytic.e_tot, embedded.e_tot, atol=1.0e-7)
    assert len(analytic.e_history) == len(embedded.e_history)
    assert embedded.orbital_rdm_last_info["core_mode"] == "embedded"
    assert embedded.orbital_rdm_last_info["rdm2_shape"] == (3, 3, 3, 3)
    assert analytic.orbital_rdm_last_info["core_mode"] == "analytic"
    assert analytic.orbital_rdm_last_info["rdm2_shape"] == (2, 2, 2, 2)
    assert (
        analytic.orbital_rdm_last_info["rdm_bytes"]
        < embedded.orbital_rdm_last_info["rdm_bytes"]
    )


def test_state_averaged_su2_dmrgscf_uses_active_only_core_rdms():
    mf = _lih_rhf()
    common = dict(
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=1,
        symmetry="su2",
        orbital_rdm_algorithm="npdm",
        init_guess="hf",
        verbose=0,
    )
    run_options = dict(
        nstates=2,
        weights=[0.5, 0.5],
        nsweeps=3,
        mixer_zero_block_noise_scale=0.0,
        require_conv=False,
        warm_start_dmrg=False,
    )
    embedded = DMRGSCF(
        mf,
        orbital_core_mode="embedded",
        **common,
    ).run(**run_options)
    analytic = DMRGSCF(
        mf,
        orbital_core_mode="analytic",
        **common,
    ).run(**run_options)

    np.testing.assert_allclose(analytic.e_tot, embedded.e_tot, atol=1.0e-5)
    assert analytic.orbital_rdm_last_info["core_mode"] == "analytic"
    assert analytic.orbital_rdm_last_info["rdm2_shape"] == (2, 2, 2, 2)


def test_factorized_su2_dmrgscf_uses_active_only_core_rdms():
    mf = _lih_factor_rhf()
    common = dict(
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=4,
        symmetry="su2",
        orbital_rdm_algorithm="npdm",
        integral_backend="ri",
        init_guess="hf",
        verbose=0,
    )
    run_options = dict(
        nstates=1,
        nsweeps=3,
        mixer_zero_block_noise_scale=0.0,
        require_conv=False,
        warm_start_dmrg=False,
    )
    embedded = DMRGSCF(
        mf,
        orbital_core_mode="embedded",
        **common,
    ).run(**run_options)
    analytic = DMRGSCF(
        mf,
        orbital_core_mode="analytic",
        **common,
    ).run(**run_options)

    assert analytic.orbital_integral_representation == "factors"
    np.testing.assert_allclose(analytic.e_tot, embedded.e_tot, atol=1.0e-7)
    assert analytic.orbital_rdm_last_info["rdm2_shape"] == (2, 2, 2, 2)


def test_state_averaged_su2_dmrgscf_preserves_su2_solver():
    mf = _h2_rhf()
    mc = DMRGSCF(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=1,
        symmetry="su2",
        init_guess="hf",
        verbose=0,
    )
    mc.run(
        nstates=2,
        weights=[0.5, 0.5],
        nsweeps=2,
        mixer_zero_block_noise_scale=0.0,
    )

    assert mc.dmrg.backend == "su2"
    assert mc.spin_purification is False
    assert mc.casci.spin_purification is False
    assert mc.dmrg_conv_tol == 1.0e-7
    assert mc.macro_converged is True
    assert mc.solver_converged == mc.dmrg.converged
    assert mc.converged == (mc.macro_converged and mc.solver_converged)
    assert mc.macro_iterations == 1
    assert mc.states == mc.dmrg.states
    assert len(mc.dmrg.states) == 2
    np.testing.assert_allclose(
        mc.e_tot,
        [-1.137275943783, -0.169291740911],
        atol=1.0e-8,
    )


def test_state_averaged_su2_dmrgscf_requires_final_inner_convergence():
    mf = _h2_rhf()
    mc = DMRGSCF(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=1,
        symmetry="su2",
        init_guess="hf",
        verbose=0,
    )

    with TestCase().assertRaisesRegex(
        RuntimeError,
        "active-space DMRG did not converge",
    ):
        mc.run(
            nstates=2,
            weights=[0.5, 0.5],
            nsweeps=1,
            conv_tol=-1.0,
            mixer_zero_block_noise_scale=0.0,
        )

    assert mc.macro_converged is True
    assert mc.solver_converged is False
    assert mc.converged is False


def test_state_averaged_su2_dmrgscf_builds_general_orbital_rdms():
    mf = _h4_rhf()
    mc = DMRGSCF(
        mf,
        ncas=4,
        nelecas=4,
        D=40,
        max_cycles=1,
        symmetry="su2",
        orbital_rdm_algorithm="npdm",
        init_guess="cid",
        verbose=0,
    )
    mc.run(
        nstates=2,
        weights=[0.5, 0.5],
        nsweeps=4,
        mixer_zero_block_noise_scale=0.0,
        require_conv=False,
    )

    assert mc.dmrg.backend == "su2"
    assert mc.spin_purification is False
    assert len(mc.states) == 2
    for state_id, energy in enumerate(mc.e_tot):
        dm1, dm2 = mc.casci.make_orbital_rdm12(state_id)
        np.testing.assert_allclose(np.trace(dm1), 4.0, atol=1.0e-10)
        np.testing.assert_allclose(np.einsum("pprr->", dm2), 12.0, atol=1.0e-10)
        reconstructed = (
            mc.casci.e_core
            + np.einsum("pq,qp->", mc.casci.h1e[0], dm1)
            + 0.5 * np.einsum("pqrs,pqrs->", mc.casci.h2e[0, 0], dm2)
        )
        np.testing.assert_allclose(reconstructed, energy, atol=1.0e-9)

    reference = CASCI(
        mf,
        ncas=4,
        nelecas=4,
        spin=0,
        verbose=0,
    ).run(
        nstates=1,
        mo_coeff=mc.mo_coeff,
        method="direct_ci",
    )
    dm1, dm2 = mc.casci.make_orbital_rdm12(0)
    reference_dm1, reference_dm2 = reference.make_rdm12(0)
    np.testing.assert_allclose(dm1, reference_dm1, atol=1.0e-10)
    np.testing.assert_allclose(
        dm2,
        _eightfold_eri_average(reference_dm2),
        atol=1.0e-10,
    )


def test_su2_npdm_orbital_rdms_match_response_backend():
    mf = _h4_rhf()
    mc = DMRGSCF(
        mf,
        ncas=4,
        nelecas=4,
        D=40,
        max_cycles=1,
        symmetry="su2",
        orbital_rdm_algorithm="npdm",
        init_guess="cid",
        verbose=0,
    )
    mc.run(
        nstates=1,
        nsweeps=4,
        mixer_zero_block_noise_scale=0.0,
        require_conv=False,
    )

    assert mc.orbital_rdm_algorithm == "npdm"
    assert mc.casci.orbital_rdm_algorithm == "npdm"
    npdm_dm1, npdm_dm2 = mc.casci.make_orbital_rdm12(0)
    assert mc.casci.orbital_rdm_last_info["representation"] == "spin_component_mps"
    assert mc.casci.orbital_rdm_last_info["component_mps_bytes"] > 0
    mc.casci.orbital_rdm_algorithm = "response"
    response_dm1, response_dm2 = mc.casci.make_orbital_rdm12(0)

    np.testing.assert_allclose(npdm_dm1, response_dm1, atol=1.0e-10)
    np.testing.assert_allclose(npdm_dm2, response_dm2, atol=1.0e-10)


def test_lih_su2_npdm_orbital_rdms_match_response_backend():
    mc = DMRGSCF(
        _lih_rhf(),
        ncas=4,
        nelecas=4,
        D=24,
        max_cycles=1,
        symmetry="su2",
        orbital_rdm_algorithm="response",
        init_guess="hf",
        verbose=0,
    )
    mc.run(
        nstates=1,
        nsweeps=4,
        mixer_zero_block_noise_scale=0.0,
        require_conv=False,
        warm_start_dmrg=False,
    )

    response_dm1, response_dm2 = mc.casci.make_orbital_rdm12(0)
    mc.casci.orbital_rdm_algorithm = "npdm"
    npdm_dm1, npdm_dm2 = mc.casci.make_orbital_rdm12(0)

    np.testing.assert_allclose(npdm_dm1, response_dm1, atol=1.0e-10)
    np.testing.assert_allclose(npdm_dm2, response_dm2, atol=1.0e-10)


def test_su2_npdm_with_frozen_core_matches_response_backend():
    mc = DMRGSCF(
        _lih_rhf(),
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=1,
        symmetry="su2",
        orbital_rdm_algorithm="response",
        init_guess="hf",
        verbose=0,
    )
    mc.run(
        nstates=1,
        nsweeps=3,
        mixer_zero_block_noise_scale=0.0,
        require_conv=False,
        warm_start_dmrg=False,
    )

    response_dm1, response_dm2 = mc.casci.make_orbital_rdm12(0, with_core=True)
    mc.casci.orbital_rdm_algorithm = "npdm"
    npdm_dm1, npdm_dm2 = mc.casci.make_orbital_rdm12(0, with_core=True)

    assert npdm_dm1.shape == (3, 3)
    assert npdm_dm2.shape == (3, 3, 3, 3)
    np.testing.assert_allclose(npdm_dm1, response_dm1, atol=1.0e-10)
    np.testing.assert_allclose(npdm_dm2, response_dm2, atol=1.0e-10)


def test_qchem_su2_sweep_measure_prefers_objective_residual():
    sweep_result = {
        "updates": [
            {"trunc_err": 0.0, "local_objective": {"metric": 2.5e-3}},
            {"trunc_err": 0.0, "local_objective": {"residual": 1.0e-2}},
        ]
    }

    assert _qchem_sweep_measure(sweep_result) == 1.0e-2


def test_dmrgscf_keeps_spin_penalty_as_an_opt_in_option():
    mc = DMRGSCF(
        _h2_rhf(),
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=1,
        verbose=0,
    )

    assert mc.spin_purification is False
    assert mc.fix_spin(ss=0.0, shift=0.2) is mc
    assert mc.spin_purification is True
    assert mc.ss == 0.0
    assert mc.shift == 0.2


def test_su2_dmrgscf_uses_factorized_orbital_optimization_integrals():
    mf = _h2_factor_rhf()

    def reject_dense_orbital_integrals(*_args, **_kwargs):
        raise AssertionError("factor-only co-DMRG must not form the dense MO ERI")

    mf.get_eri_mo = reject_dense_orbital_integrals
    mc = DMRGSCF(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=1,
        symmetry="su2",
        orbital_rdm_algorithm="npdm",
        integral_backend="ri",
        init_guess="hf",
        verbose=0,
    )
    mc.run(
        nstates=1,
        nsweeps=2,
        mixer_zero_block_noise_scale=0.0,
        require_conv=False,
    )

    assert mc.orbital_integral_representation == "factors"
    assert mc.orbital_integral_shape[0] == mf.eri_factors.shape[0]
    assert mc.orbital_integral_shape[1:] == (mf.nao, mf.nao)


def test_su2_dmrgscf_macro_callback_exposes_restartable_state():
    events = []
    mc = DMRGSCF(
        _h2_rhf(),
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=1,
        symmetry="su2",
        orbital_rdm_algorithm="npdm",
        init_guess="hf",
        verbose=0,
    )
    mc.run(
        nstates=1,
        nsweeps=2,
        mixer_zero_block_noise_scale=0.0,
        require_conv=False,
        macro_callback=events.append,
    )

    assert len(events) == 1
    event = events[0]
    assert event["macro"] == 1
    assert event["accepted"] is True
    assert event["mo_coeff"].shape == mc.mo_coeff.shape
    np.testing.assert_allclose(event["energy_history"][-1], event["energy"])
    assert event["diagnostics"]["macro"] == 1
    state = event["casci"].export_ground_state(state=0)
    restored = pickle.loads(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
    assert len(restored) == len(state)


def test_local_comparison_runs_su2_response_and_npdm_dmrgscf():
    response, npdm = _run_co_dmrg_comparison(
        _h2_rhf(),
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=1,
        nstates=2,
        weights=[0.5, 0.5],
        nsweeps=2,
        init_guess="hf",
    )

    assert response.symmetry == ["charge", "su2"]
    assert npdm.symmetry == ["charge", "su2"]
    assert response.orbital_rdm_algorithm == "response"
    assert npdm.orbital_rdm_algorithm == "npdm"
    assert response.spin_purification is False
    assert npdm.spin_purification is False
    assert np.asarray(response.e_tot).reshape(-1).shape == (2,)
    assert np.asarray(npdm.e_tot).reshape(-1).shape == (2,)
    np.testing.assert_allclose(response.e_tot, npdm.e_tot, atol=1.0e-9)

    output = StringIO()
    with redirect_stdout(output):
        _print_co_dmrg_comparison(response, npdm, weights=[0.5, 0.5])
    report = output.getvalue()
    assert "Energy history" in report
    assert "Macroiterations used" in report
    assert "Final root energies" in report
    assert "response - NPDM" in report


def test_local_comparison_reports_active_only_core_time_and_memory():
    embedded, analytic = _run_core_mode_comparison(
        _lih_rhf(),
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=1,
        nstates=1,
        weights=[1.0],
        nsweeps=3,
        init_guess="hf",
    )

    assert embedded.orbital_core_mode == "embedded"
    assert analytic.orbital_core_mode == "analytic"
    np.testing.assert_allclose(embedded.e_tot, analytic.e_tot, atol=1.0e-5)
    assert (
        analytic.orbital_rdm_last_info["rdm_bytes"]
        < embedded.orbital_rdm_last_info["rdm_bytes"]
    )
    assert embedded.comparison_wall_time_s > 0.0
    assert analytic.comparison_wall_time_s > 0.0

    output = StringIO()
    with redirect_stdout(output):
        _print_core_mode_comparison(embedded, analytic, weights=[1.0])
    report = output.getvalue()
    assert "Before: embedded core RDM" in report
    assert "After: active-only RDM + analytic core" in report
    assert "Optimizer RDM shapes" in report
    assert "RDM storage reduction" in report
    assert "embedded - analytic" in report


def _run_co_dmrg_comparison(
    mf,
    *,
    ncas,
    nelecas,
    D,
    max_cycles,
    nstates,
    weights,
    nsweeps,
    init_guess,
    macro_tol=1.0e-6,
    dmrg_conv_tol=1.0e-7,
    require_conv=False,
    warm_start_dmrg=False,
):
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (nstates,):
        raise ValueError("weights must contain one value per requested state.")
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("state-average weights must sum to one.")

    driver_options = dict(
        nstates=nstates,
        weights=weights.tolist(),
        nsweeps=nsweeps,
        conv_tol=dmrg_conv_tol,
        mixer_zero_block_noise_scale=0.0,
        require_conv=require_conv,
        warm_start_dmrg=warm_start_dmrg,
    )
    solver_options = dict(
        ncas=ncas,
        nelecas=nelecas,
        D=D,
        max_cycles=max_cycles,
        macro_tol=macro_tol,
        dmrg_conv_tol=dmrg_conv_tol,
        init_guess=init_guess,
        verbose=0,
    )

    response = DMRGSCF(
        mf,
        symmetry="su2",
        orbital_rdm_algorithm="response",
        **solver_options,
    )
    started = perf_counter()
    response.run(**driver_options)
    response.comparison_wall_time_s = perf_counter() - started

    npdm = DMRGSCF(
        mf,
        symmetry="su2",
        orbital_rdm_algorithm="npdm",
        **solver_options,
    )
    started = perf_counter()
    npdm.run(**driver_options)
    npdm.comparison_wall_time_s = perf_counter() - started
    return response, npdm


def _run_with_timing(solver, driver_options):
    started = perf_counter()
    solver.run(**driver_options)
    solver.comparison_wall_time_s = perf_counter() - started
    return solver


def _run_core_mode_comparison(
    mf,
    *,
    ncas,
    nelecas,
    D,
    max_cycles,
    nstates,
    weights,
    nsweeps,
    init_guess,
    macro_tol=1.0e-6,
    dmrg_conv_tol=1.0e-7,
    require_conv=False,
    warm_start_dmrg=False,
):
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (nstates,):
        raise ValueError("weights must contain one value per requested state.")
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("state-average weights must sum to one.")
    driver_options = dict(
        nstates=nstates,
        weights=weights.tolist(),
        nsweeps=nsweeps,
        conv_tol=dmrg_conv_tol,
        mixer_zero_block_noise_scale=0.0,
        require_conv=require_conv,
        warm_start_dmrg=warm_start_dmrg,
    )
    solver_options = dict(
        ncas=ncas,
        nelecas=nelecas,
        D=D,
        max_cycles=max_cycles,
        macro_tol=macro_tol,
        dmrg_conv_tol=dmrg_conv_tol,
        symmetry="su2",
        orbital_rdm_algorithm="npdm",
        init_guess=init_guess,
        verbose=0,
    )
    embedded = _run_with_timing(
        DMRGSCF(mf, orbital_core_mode="embedded", **solver_options),
        driver_options,
    )
    analytic = _run_with_timing(
        DMRGSCF(mf, orbital_core_mode="analytic", **solver_options),
        driver_options,
    )
    return embedded, analytic


def _energy_vector(energy):
    return np.asarray(energy, dtype=float).reshape(-1)


def _print_method_result(label, solver, weights):
    weights = np.asarray(weights, dtype=float)
    print(f"\n{label}")
    print("-" * len(label))
    print(f"Symmetry labels: {solver.symmetry}")
    print(f"Orbital RDM algorithm: {solver.orbital_rdm_algorithm}")
    print(f"Inactive-core treatment: {solver.orbital_core_mode}")
    print(f"Spin penalty enabled: {solver.spin_purification}")
    print("Energy history:")
    for step, energy in enumerate(solver.e_history):
        roots = _energy_vector(energy)
        average = float(np.dot(weights, roots))
        name = "initial" if step == 0 else f"macro {step}"
        print(
            f"  {name:>8}: roots={np.array2string(roots, precision=12)}, "
            f"weighted={average:.12f}"
        )
    print(f"Macroiterations used: {solver.macro_iterations}")
    print(f"Macro converged: {solver.macro_converged}")
    print(f"Final DMRG converged: {solver.solver_converged}")
    print(
        "Final root energies: "
        f"{np.array2string(_energy_vector(solver.e_tot), precision=12)}"
    )
    if hasattr(solver, "comparison_wall_time_s"):
        print(f"Wall time: {solver.comparison_wall_time_s:.2f} s")
    rdm_info = getattr(solver, "orbital_rdm_last_info", None)
    if rdm_info:
        if "wall_time_s" in rdm_info:
            print(
                f"Last orbital-RDM build: {rdm_info['wall_time_s']:.4f} s, "
                f"component MPS="
                f"{rdm_info.get('component_mps_bytes', 0) / 2**20:.3f} MiB"
            )
        print(
            f"Optimizer RDM shapes: {rdm_info['rdm1_shape']} / "
            f"{rdm_info['rdm2_shape']}; stored="
            f"{rdm_info['rdm_bytes'] / 2**20:.6f} MiB"
        )


def _print_co_dmrg_comparison(response, npdm, *, weights):
    weights = np.asarray(weights, dtype=float)
    _print_method_result("SU(2) response co-DMRG", response, weights)
    _print_method_result("SU(2) NPDM co-DMRG", npdm, weights)

    response_energy = _energy_vector(response.e_tot)
    npdm_energy = _energy_vector(npdm.e_tot)
    if response_energy.shape != npdm_energy.shape:
        raise ValueError("Response and NPDM calculations returned different root counts.")
    difference = response_energy - npdm_energy
    print("\nFinal energy comparison")
    print("-----------------------")
    for root, (response_root, npdm_root, delta) in enumerate(
        zip(response_energy, npdm_energy, difference)
    ):
        print(
            f"Root {root}: response={response_root:.12f}, NPDM={npdm_root:.12f}, "
            f"response - NPDM={delta:+.12e}"
        )
    print(
        "Weighted response - NPDM: "
        f"{float(np.dot(weights, difference)):+.12e}"
    )
    print(f"Maximum absolute root difference: {np.max(np.abs(difference)):.12e}")


def _print_core_mode_comparison(embedded, analytic, *, weights):
    weights = np.asarray(weights, dtype=float)
    _print_method_result("Before: embedded core RDM", embedded, weights)
    _print_method_result(
        "After: active-only RDM + analytic core",
        analytic,
        weights,
    )
    embedded_energy = _energy_vector(embedded.e_tot)
    analytic_energy = _energy_vector(analytic.e_tot)
    difference = embedded_energy - analytic_energy
    before_bytes = embedded.orbital_rdm_last_info["rdm_bytes"]
    after_bytes = analytic.orbital_rdm_last_info["rdm_bytes"]
    reduction = before_bytes / after_bytes if after_bytes else np.inf

    print("\nBefore/after comparison")
    print("-----------------------")
    for root, (old_root, new_root, delta) in enumerate(
        zip(embedded_energy, analytic_energy, difference)
    ):
        print(
            f"Root {root}: embedded={old_root:.12f}, analytic={new_root:.12f}, "
            f"embedded - analytic={delta:+.12e}"
        )
    print(
        "Weighted embedded - analytic: "
        f"{float(np.dot(weights, difference)):+.12e}"
    )
    print(
        f"RDM storage reduction: {before_bytes / 2**20:.6f} -> "
        f"{after_bytes / 2**20:.6f} MiB ({reduction:.2f}x smaller)"
    )


def main():
    # Add, remove, or edit entries here. Every case uses SU(2) NPDM twice; only
    # the inactive-core treatment changes (old embedded RDM versus active-only).
    cases = [
        {
            "name": "H2",
            "atom": "H 0 0 0; H 0 0 1.4",
            "unit": "bohr",
            "basis": "sto-3g",
            "ncas": 2,
            "nelecas": 2,
            "D": 8,
        },
        {
            "name": "LiH",
            "atom": "Li 0 0 0; H 0 0 3.0",
            "unit": "bohr",
            "basis": "sto-3g",
            "ncas": 4,
            "nelecas": 4,
            "D": 24,
        },
        {
            "name": "H4 chain",
            "atom": "H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
            "unit": "bohr",
            "basis": "sto-3g",
            "ncas": 4,
            "nelecas": 4,
            "D": 32,
        },
        {
            "enabled": True,
            "name": "LiF",
            "atom": "Li 0 0 0; F 0 0 3.0",
            "unit": "bohr",
            "basis": "6-311g",
            "ncas": 6,
            "nelecas": 6,
            "D": 200,
        },
    ]
    max_macroiterations = 8
    nstates = 1
    weights = np.array([1.0])
    nsweeps = 10
    init_guess = "hf"

    suite_started = perf_counter()
    for case in cases:
        if not case.get("enabled", True):
            continue
        mol = Molecule(
            atom=case["atom"],
            unit=case["unit"],
            basis=case["basis"],
        )
        mol.build(
            driver="builtin",
            eri="dense",
            aosym="s1",
            options={"eri_backend": "auto"},
        )
        mf = RHF(mol).run()

        print("\n" + "=" * 72)
        print(f"{case['name']}: {case['atom']} ({case['unit']})")
        print(
            f"Basis={case['basis']}, CAS=({case['ncas']}o, {case['nelecas']}e), "
            f"D={case['D']}, states/weights={nstates}/{weights.tolist()}"
        )
        embedded, analytic = _run_core_mode_comparison(
            mf,
            ncas=case["ncas"],
            nelecas=case["nelecas"],
            D=case["D"],
            max_cycles=max_macroiterations,
            nstates=nstates,
            weights=weights,
            nsweeps=nsweeps,
            init_guess=init_guess,
            require_conv=False,
            warm_start_dmrg=False,
        )
        _print_core_mode_comparison(embedded, analytic, weights=weights)
    print(f"\nSuite wall time: {perf_counter() - suite_started:.2f} s")


if __name__ == "__main__":
    main()
