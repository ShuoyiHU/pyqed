# Allocation-failure recovery checkpoint — 2026-10-05

The native reduced backends now classify MemoryError alongside numerical failures. An ordinary open-chain site transaction restores all tensors and sector allocations before propagating its failure. The sweep catches that failure, rebuilds moving environments, records a rejected unresolved update, and continues at the next site. Ring ordinary updates and gauge shifts now use the same exception policy as ring CBE/two-site. Failed expansion/compression candidates retain the separately computed one-site baseline, with the next call free to try CBE/two-site again.

This covers recoverable Python allocation failures while an incumbent and its energy can still be evaluated. It cannot recover an operating-system kill or guarantee progress if even the checkpoint, baseline, or restored environments cannot be allocated. No physical metric, convergence tolerance, bond allocation, or environment rank is reduced to avoid allocation errors.

Six targeted tests failed before the change and passed afterward. They cover partially written tensors and sector metadata in open/ring one-site updates and ring gauges, plus allocation failure in the actual shared compression call reached by CBE and two-site. The latter two also retry successfully after removing the fault, lowering energy by more than 1e-4 from the ordinary baseline. The earlier proposed workspace-limit tests were incorrect: nonlinear compression deliberately falls back to ALS when its workspace estimate exceeds the configured budget. The tests now inject MemoryError at compression instead, preserving that intended ALS behavior.

Validation (overlapping counts, not a unique total):

- Six targeted tests: 6 passed in 3.88 s.
- Resource/open-update/gauge/CBE regressions: 56 passed in 18.47 s.
- Ring one-site/CBE/two-site failure and workspace regressions: 18 passed, 44 deselected in 9.80 s.
- git diff --check passed.

Commands use the existing main-checkout .venv-1 Python with PYTHONPATH=., bytecode disabled, temporary Numba/Matplotlib caches and single BLAS/OpenMP threads. Exact pytest selections:

```sh
python -m pytest -p no:cacheprovider -q tests/test_letta_resource_recovery.py
python -m pytest -p no:cacheprovider -q tests/test_letta_resource_recovery.py tests/test_letta_reduced_updates.py tests/test_letta_reduced_sweep_recovery.py tests/test_letta_reduced_cbe.py
python -m pytest -p no:cacheprovider -q tests/test_letta_ring_sweeps.py tests/test_letta_ring_cbe.py tests/test_letta_ring_two_site.py -k 'fail or recovery or workspace'
```

The full public API/model/topology run is still pending. This checkpoint is not a completion claim for the support matrix or the full implementation note. The optional performance phase has not begun.
