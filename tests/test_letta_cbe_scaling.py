import numpy as np
import pytest

from pyqed._letta_one_site_opt.benchmarks.cbe_scaling import (
    run_scaling_profile,
)


@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_strict_shrewd_graph_has_one_site_scaling(direction):
    report = run_scaling_profile(
        bond_dimensions=(2, 4, 8),
        physical_dimensions=(2, 3, 4),
        mpo_widths=(8, 16, 32),
        direction=direction,
    )
    exponents = report["exponents"]

    assert report["proof"]["pair_actions"] == 0
    assert report["proof"]["pair_metrics"] == 0
    assert report["proof"]["merged_pairs"] == 0
    assert (
        exponents["bond"]["strict_selector"]
        <= exponents["bond"]["one_site_action"] + 0.05
    )
    assert (
        exponents["bond"]["strict_selector_with_svd"]
        <= exponents["bond"]["one_site_action"] + 0.05
    )
    assert abs(
        exponents["physical"]["strict_selector"]
        - exponents["physical"]["one_site_action"]
    ) < 0.15
    assert (
        exponents["physical"]["pair_action"]
        - exponents["physical"]["strict_selector_with_svd"]
        > 0.7
    )
    assert abs(
        exponents["mpo"]["strict_selector"]
        - exponents["mpo"]["one_site_action"]
    ) < 0.05
    assert np.isclose(exponents["mpo"]["strict_selector"], 1.0)
    assert abs(
        exponents["mpo"]["strict_selector_with_svd"]
        - exponents["mpo"]["one_site_action"]
    ) < 0.08
    for profile in (
        report["bond_profile"],
        report["physical_profile"],
        report["mpo_profile"],
    ):
        assert all(
            point["strict_selector"]["largest_live_tensor"]
            <= point["one_site_action"]["largest_live_tensor"]
            for point in profile
        )
