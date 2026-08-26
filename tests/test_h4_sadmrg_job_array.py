import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "_local_tests"
    / "_yujuan_tddmrg"
    / "_July21th_H4_sadmrg_pec"
    / "cluster_nz48"
    / "h4_gdvr_sadmrg_array.py"
)
if not SCRIPT.exists():
    SCRIPT = Path(__file__).resolve().parents[1] / "h4_gdvr_sadmrg_array.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("h4_gdvr_sadmrg_array", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_job_array_covers_the_15_by_15_grid_exactly():
    module = _load_module()
    points = [module.task_point(index) for index in range(225)]

    assert points[0][:4] == (0, 0, -0.4375, -0.4375)
    assert points[-1][:4] == (14, 14, 0.4375, 0.4375)
    assert len({(q1, q2) for _, _, q1, q2, _ in points}) == 225
    np.testing.assert_allclose(points[112][2:4], [0.0, 0.0])


def test_sweep_history_upserts_records_on_resume(tmp_path):
    module = _load_module()
    path = tmp_path / "sweep_history.json"

    module.upsert_sweep_record(path, {"half_sweep": 0, "s2": [0.1]})
    module.upsert_sweep_record(path, {"half_sweep": 1, "s2": [0.01]})
    module.upsert_sweep_record(path, {"half_sweep": 1, "s2": [0.001]})

    records = json.loads(path.read_text())
    assert [record["half_sweep"] for record in records] == [0, 1]
    assert records[-1]["s2"] == [0.001]


def test_existing_task_configuration_must_match(tmp_path):
    module = _load_module()
    path = tmp_path / "config.json"
    config = {"task_index": 7, "Lz": 6.0, "Nz": 48}

    module.ensure_task_config(path, config)
    module.ensure_task_config(path, dict(config))

    try:
        module.ensure_task_config(path, {**config, "Nz": 47})
    except ValueError as exc:
        assert "different settings" in str(exc)
    else:
        raise AssertionError("mismatched resume configuration was accepted")


def test_normalized_root_expectations_accept_dense_sweep_roots():
    module = _load_module()
    root = [np.array([[[1.0], [0.0]]])]
    identity = [np.eye(2).reshape(1, 1, 2, 2)]

    values = module.normalized_root_expectations([root], identity)

    np.testing.assert_allclose(values, [1.0])
