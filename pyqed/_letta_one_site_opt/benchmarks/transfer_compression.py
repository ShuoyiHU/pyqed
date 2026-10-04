"""Audit segment SVD on initial/optimized states before enabling sweep use.

The full norm-transfer matrix is a bounded *reference only*. The compressor
uses matrix-free contractions. No approximate environments enter optimization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pyqed._letta_one_site_opt import LatticeLETTA, LETTADMROptions, letta_dmrg
from pyqed._letta_two_site_opt import (
    IdentityPairEnvironmentCache, LETTAPairLayout, LETTATwoSiteOptions, letta_two_site_dmrg,
)
from .condensed_models import build_model


def profile_state(state, mpo, *, ranks=(4, 8, 16), tolerance=1e-8,
                  max_dense_elements=1_000_000):
    cache = IdentityPairEnvironmentCache(state)
    left, right = cache.build_left_environments(), cache.build_right_environments()
    norm = np.asarray(left[-1]).item()
    energy = state.expectation(mpo)
    segments = {(1, state.nsites - 1)}
    width = state.lattice_shape[-1]
    if 0 < width < state.nsites - width:
        segments.add((width, state.nsites - width))
    reports = []
    for start, stop in sorted(segments):
        transfer = cache.segment_transfer(start, stop)
        report = {"start": start, "stop": stop, "shape": transfer.shape, "candidates": []}
        dense = None
        if np.prod(transfer.shape) <= max_dense_elements:
            # Stream basis vectors, avoiding an n x n identity allocation.
            dense = np.empty(transfer.shape, dtype=transfer.dtype)
            basis = np.zeros(transfer.shape[1], dtype=transfer.dtype)
            for col in range(transfer.shape[1]):
                basis[col] = 1
                dense[:, col] = transfer @ basis
                basis[col] = 0
            singular_values = np.linalg.svd(dense, compute_uv=False)
            tail = np.r_[np.cumsum(singular_values[::-1]**2)[::-1], 0.]
            tail = np.sqrt(tail / tail[0]) if tail[0] else np.zeros_like(tail)
            report["singular_values_relative"] = (singular_values / singular_values[0]).tolist()
            report["minimum_rank_for_relative_frobenius_error"] = {
                str(tol): int(np.flatnonzero(tail <= tol)[0]) for tol in (1e-2, 1e-4, 1e-8)
            }
        for rank in sorted({min(rank, min(transfer.shape)) for rank in ranks}):
            began = perf_counter()
            compressed = transfer.compress(rank, tolerance=tolerance)
            candidate = dict(compressed.diagnostics, build_seconds=perf_counter() - began)
            if dense is not None:
                approximation = (compressed.candidate.u * compressed.candidate.singular_values) @ compressed.candidate.vh
                candidate["actual_relative_frobenius_error"] = float(np.linalg.norm(dense - approximation) / np.linalg.norm(dense))
                candidate["best_rank_error"] = float(tail[rank])
            # Examine the raw candidate, even when the guard rejects it.
            boundary = (compressed.candidate @ left[start].ravel()).reshape(transfer.output_shape)
            approximate_norm = np.sum(boundary * right[stop])
            candidate["raw_relative_norm_error"] = float(abs(approximate_norm - norm) / abs(norm))
            candidate["raw_norm_imaginary_fraction"] = float(abs(approximate_norm.imag) / abs(norm))
            candidate["energy_shift_from_norm_only"] = (
                float(np.real(energy * norm / approximate_norm) - energy)
                if abs(approximate_norm) > np.finfo(float).tiny else None)
            if stop + 2 <= state.nsites:
                layout = LETTAPairLayout.from_state(state, stop)
                exact_metric = cache.effective_pair_metric(left[stop], right[stop + 2], layout)
                approx_metric = cache.effective_pair_metric(boundary, right[stop + 2], layout)
                exact_blocks, approx_blocks = np.stack(exact_metric.blocks), np.stack(approx_metric.blocks)
                candidate["raw_pair_metric_relative_error"] = float(
                    np.linalg.norm(approx_blocks - exact_blocks) / np.linalg.norm(exact_blocks))
                hermitian = .5 * (approx_blocks + approx_blocks.conj().swapaxes(-1, -2))
                exact_hermitian = .5 * (exact_blocks + exact_blocks.conj().swapaxes(-1, -2))
                exact_eigenvalues = np.linalg.eigvalsh(exact_hermitian)
                candidate["exact_pair_metric_min_eigenvalue"] = float(exact_eigenvalues.min())
                candidate["raw_pair_metric_min_eigenvalue"] = float(np.linalg.eigvalsh(hermitian).min())
                candidate["raw_pair_metric_min_eigenvalue_relative_to_exact_scale"] = float(
                    np.linalg.eigvalsh(hermitian).min() / exact_eigenvalues.max())
                candidate["raw_pair_metric_hermiticity_error"] = float(
                    np.linalg.norm(approx_blocks - hermitian) / np.linalg.norm(exact_blocks))
            for name, operator in (("exact", transfer), ("candidate", compressed.candidate)):
                vector = left[start].ravel()
                operator @ vector
                began = perf_counter()
                for _ in range(10):
                    operator @ vector
                candidate[f"{name}_application_seconds"] = (perf_counter() - began) / 10
            saving = candidate["exact_application_seconds"] - candidate["candidate_application_seconds"]
            candidate["estimated_reuse_to_amortize_build"] = (
                int(np.ceil(candidate["build_seconds"] / saving)) if saving > 0 else None)
            report["candidates"].append(candidate)
        reports.append(report)
    return {"energy": float(energy), "segments": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="bose_hubbard", choices=("bose_hubbard", "ising", "heisenberg"))
    parser.add_argument("--shape", type=int, nargs=2, default=(3, 3))
    parser.add_argument("--bond-dim", type=int, default=3)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--sweeps", type=int, default=6)
    parser.add_argument("--methods", nargs="+", choices=("one-site", "cbe", "two-site"), default=("one-site",))
    parser.add_argument("--ranks", type=int, nargs="+", default=(4, 8, 16))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sweeps <= 0 or any(rank <= 0 for rank in args.ranks):
        parser.error("sweeps and ranks must be positive")
    dimension = "1d" if args.shape[0] == 1 else "2d"
    model = build_model(args.model, dimension=dimension,
                        size=args.shape[1] if dimension == "1d" else tuple(args.shape))
    initial = LatticeLETTA.random(model.lattice_shape, physical_dim=model.physical_dim,
                                  bond_dim=args.bond_dim, seed=args.seed)
    report = {"parameters": dict(vars(args), output=str(args.output)), "states": {}}

    def record(name, state, seconds=0.):
        report["states"][name] = dict(profile_state(state, model.mpo, ranks=args.ranks),
                                      optimization_seconds=seconds)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(name, "energy", report["states"][name]["energy"], flush=True)

    record("initial", initial)
    for method in args.methods:
        began = perf_counter()
        if method == "two-site":
            result = letta_two_site_dmrg(model.mpo, state=initial, bond_dim=args.bond_dim,
                                         options=LETTATwoSiteOptions(max_sweeps=args.sweeps, tolerance=1e-14))
        else:
            result = letta_dmrg(model.mpo, state=initial,
                                options=LETTADMROptions(max_sweeps=args.sweeps, tolerance=1e-14,
                                                        cbe_enabled=method == "cbe", cbe_selector="shrewd"))
        record(method, result.state, perf_counter() - began)


if __name__ == "__main__":
    main()
