from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "pyqed/_letta_one_site_opt/_letta_for_3d/examples/count_tfi_variational_parameters.py"
)
SPEC = spec_from_file_location("count_tfi_variational_parameters", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
COUNTER = module_from_spec(SPEC)
SPEC.loader.exec_module(COUNTER)

_rank_capped_bonds = COUNTER._rank_capped_bonds
parameter_count_records = COUNTER.parameter_count_records


def test_mps_d32_bond_ranks_are_boundary_capped() -> None:
    bonds = _rank_capped_bonds((2,) * 27, 32)
    assert bonds[:5] == (2, 4, 8, 16, 32)
    assert bonds[-5:] == (32, 16, 8, 4, 2)


def test_requested_stored_parameter_counts() -> None:
    expected = {
        (3, "LETTA", 1): 250,
        (3, "LETTA", 4): 3748,
        (3, "LETTA", 6): 8356,
        (3, "MPS", 1): 54,
        (3, "MPS", 4): 776,
        (3, "MPS", 8): 2856,
        (3, "MPS", 16): 10408,
        (3, "MPS", 32): 37544,
        (6, "LETTA", 1): 550,
        (6, "LETTA", 4): 8548,
        (6, "LETTA", 6): 19156,
        (6, "MPS", 1): 108,
        (6, "MPS", 4): 1640,
        (6, "MPS", 8): 6312,
        (6, "MPS", 16): 24232,
        (6, "MPS", 32): 92840,
        (9, "LETTA", 1): 850,
        (9, "LETTA", 4): 13348,
        (9, "LETTA", 6): 29956,
        (9, "MPS", 1): 162,
        (9, "MPS", 4): 2504,
        (9, "MPS", 8): 9768,
        (9, "MPS", 16): 38056,
        (9, "MPS", 32): 148136,
    }
    actual = {
        (record["Nx"], record["ansatz"], record["max_bond_dimension_D"]): record[
            "stored_tensor_entries"
        ]
        for record in parameter_count_records()
    }
    assert actual == expected
