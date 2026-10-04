"""Small reproducible sweep-budget comparison of QR and frontier gauges.

Run with BLAS/OpenMP thread counts set to one, from the repository root::

    PYTHONPATH=. python -m pyqed._letta_one_site_opt.benchmarks.frontier_gauge \
        --repeats 3 --output /private/tmp/frontier_gauge.json

Times include solver initialization. Energies are reported at the same maximum sweep
budget with actual sweep counts; early convergence is allowed. Coordinate checks include fallback rechecks.
"""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import warnings
from collections import Counter

import numpy as np
import scipy

from .. import LatticeLETTA, LETTADMROptions, frontier_gauge_cuts, letta_dmrg
from ..contractions import BlockDiagonalMetric
from .._letta_for_2d import transverse_field_ising_mpo


class CrossingExample(LatticeLETTA):
    dependencies = ((0, 2), (1,), (2, 3), (3, 4), (4,))

    def _build_neighborhood(self, coordinate):
        return self.dependencies[coordinate[1]]

    def copy(self):
        return type(self)(self.lattice_shape, self.physical_dim, self.tensors)


def crossing_example():
    rng = np.random.default_rng(91)
    tensors = [rng.normal(size=(1 if i == 0 else 2,) + (2,) * len(ns)
                               + (1 if i == 4 else 2,))
               for i, ns in enumerate(CrossingExample.dependencies)]
    return CrossingExample((1, 5), 2, tensors)


def run(repeats=3, sweeps=3):
    rows = []
    cases = [("internal_crossing", crossing_example())]
    cases += [(name, LatticeLETTA.random(shape, bond_dim=dimension, seed=73))
              for name, shape, dimension in (
                  ("chain_D4", (1, 8), 4), ("chain_D8", (1, 8), 8),
                  ("chain_D16", (1, 8), 16),
                  ("lattice_2x3_D3", (2, 3), 3))]
    original = BlockDiagonalMetric.coordinate_whitening
    for name, state in cases:
        mpo = transverse_field_ising_mpo(state.lattice_shape, field=.9)
        cuts = frontier_gauge_cuts(state)
        # Alternate method order to reduce warm-cache bias in reported medians.
        samples = {mode: [] for mode in ("qr", "frontier")}
        for repeat in range(repeats):
            for mode in (("qr", "frontier") if repeat % 2 == 0 else ("frontier", "qr")):
                counts = dict(checks=0, coordinate=0)

                def counted(metric, tolerance):
                    result = original(metric, tolerance)
                    counts["checks"] += 1
                    counts["coordinate"] += result is not None
                    return result

                with patch.object(BlockDiagonalMetric, "coordinate_whitening", counted):
                    started = perf_counter()
                    result = letta_dmrg(mpo, state=state, options=LETTADMROptions(
                        max_sweeps=sweeps, tolerance=1e-14, gauge_mode=mode,
                        dense_solver_threshold=1, matrix_free=True,
                    ))
                    elapsed = perf_counter() - started
                kinds = Counter(update.metric_kind for sweep in result.history for update in sweep.updates)
                compact = sum(v for k, v in kinds.items() if k != "general")
                counts["coordinate"] += compact
                counts["checks"] += compact
                physical_energy = result.state.expectation(mpo)
                if not np.isfinite(physical_energy) or abs(physical_energy - result.energy) > 1e-8:
                    raise AssertionError("cached energy differs from fresh physical contraction")
                samples[mode].append(dict(seconds=elapsed, energy=result.energy,
                                          sweeps=result.sweeps, metric_kinds=dict(kinds),
                                          canonical_environment_reuses=result.canonical_environment_reuses,
                                          canonical_metric_hits=result.canonical_metric_hits, **counts))
        for mode, values in samples.items():
            row = dict(case=name, gauge=mode, admissible_cuts=sum(c.admissible for c in cuts),
                       cuts=len(cuts), seconds=float(np.median([v["seconds"] for v in values])),
                       energy=values[-1]["energy"], sweeps=values[-1]["sweeps"], checks=values[-1]["checks"],
                       coordinate=values[-1]["coordinate"], metric_kinds=values[-1]["metric_kinds"],
                       canonical_environment_reuses=values[-1]["canonical_environment_reuses"],
                       canonical_metric_hits=values[-1]["canonical_metric_hits"], samples=values)
            rows.append(row)
            print(f'{name:20s} {mode:8s} {row["seconds"]:8.3f} s  '
                  f'E={row["energy"]:.12f}  sweeps={row["sweeps"]}  coordinates={row["coordinate"]}/{row["checks"]}', flush=True)
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                repeats=repeats, sweep_budget=sweeps, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sweeps", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.sweeps < 1:
        parser.error("repeats and sweeps must be positive")
    # Some macOS BLAS builds leave floating-point status flags set despite
    # finite results. Physical energy checks above remain explicit.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul")
        result = run(args.repeats, args.sweeps)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
