import importlib


def test_one_site_letta_code_has_dimension_independent_core_and_case_packages():
    core = importlib.import_module("pyqed._letta_one_site_opt")
    case_2d = importlib.import_module(
        "pyqed._letta_one_site_opt._letta_for_2d"
    )
    case_3d = importlib.import_module(
        "pyqed._letta_one_site_opt._letta_for_3d"
    )

    assert core.LatticeLETTA.__module__ == "pyqed._letta_one_site_opt.state"
    assert core.LatticeMPO.__module__ == "pyqed._letta_one_site_opt.operators"
    assert core.letta_dmrg.__module__ == "pyqed._letta_one_site_opt.solver"
    assert callable(case_2d.transverse_field_ising_mpo)
    assert callable(case_3d.letta_ground_state)


def test_case_packages_share_the_core_state_and_options_types():
    core = importlib.import_module("pyqed._letta_one_site_opt")
    case_2d = importlib.import_module(
        "pyqed._letta_one_site_opt._letta_for_2d"
    )
    case_3d_letta = importlib.import_module(
        "pyqed._letta_one_site_opt._letta_for_3d.letta"
    )

    assert case_2d.LatticeLETTA is core.LatticeLETTA
    assert case_2d.LETTADMROptions is core.LETTADMROptions
    assert case_3d_letta.LatticeLETTA is core.LatticeLETTA
    assert case_3d_letta.LETTADMROptions is core.LETTADMROptions
