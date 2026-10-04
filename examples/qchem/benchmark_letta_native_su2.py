"""Equivalent-state native/component SU(2) microbenchmark (not DMRG timing).

Run with PYTHONPATH=. and BLAS/OpenMP threads set to one. Compilation, boundary
builds, complete local actions, and component-only kernels are timed separately.
The comparison uses the *same* compressed MPO so MPO compression is not counted
as a spin-contraction speedup. Component expansion is included only in the
complete component action, matching the prior LETTA solver implementation.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pyqed._letta_one_site_opt import ReducedLatticeLETTA
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.reduced_frontier import ReducedFrontier
from pyqed._letta_one_site_opt.reduced_environment import ReducedEnvironmentChain
from pyqed._letta_one_site_opt.reduced_contraction import (
    CanonicalEnvironmentChain, expand_reduced_mps_site, reduce_expanded_mps_site,
)


def time_call(function, repeats=1):
    samples = []
    value = None
    for _ in range(repeats):
        start = perf_counter()
        value = function()
        samples.append(perf_counter()-start)
    return value, float(np.median(samples))


def benchmark(norb, multiplicity, repeats):
    rng = np.random.default_rng(431)
    h = rng.normal(size=(norb, norb)); h = (h+h.T)/2
    factors = rng.normal(size=(3, norb, norb)); factors = (factors+factors.transpose(0, 2, 1))/2
    eri = np.einsum('xpq,xrs->pqrs', factors, factors)
    problem = ElectronicProblem(h, eri, ((norb+1)//2, norb//2))
    state = ReducedLatticeLETTA.random((1, norb), symmetry=problem.symmetry('su2'),
        multiplets_per_sector=multiplicity, seed=7)
    hamiltonian, assembly_seconds = time_call(problem.su2_mpo)
    mpo, compilation_seconds = time_call(lambda: hamiltonian.native_mpo(state.physical_basis))
    sites = tuple(ReducedFrontier.from_state(state).to_mps(state))
    native, native_build_seconds = time_call(lambda: ReducedEnvironmentChain.build(sites, mpo))
    components = mpo.component_factors()
    reference, component_build_seconds = time_call(lambda: CanonicalEnvironmentChain.build(sites, components))
    site = norb//2
    def native_action():
        return native.local_action(site, sites[site].data)
    def component_action():
        expanded = expand_reduced_mps_site(sites[site])
        return reduce_expanded_mps_site(sites[site], reference.local_action(site, expanded))
    # Warm contraction paths and spin coefficient caches before timings.
    actual, expected = native_action(), component_action()
    absolute_error = max(float(np.max(np.abs(actual[k]-v))) for k, v in expected.items())
    np.testing.assert_allclose(native.expectation(), reference.expectation(), rtol=1e-10, atol=1e-10)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-10, atol=1e-10)
    _, native_seconds = time_call(native_action, repeats)
    _, component_seconds = time_call(component_action, repeats)
    expanded = expand_reduced_mps_site(sites[site])
    _, component_kernel_seconds = time_call(lambda: reference.local_action(site, expanded), repeats)
    native_environment_bytes = sum(a.nbytes for env in native.left+native.right for a in env.values())
    component_environment_bytes = sum(a.nbytes for a in reference.left+reference.right)
    return dict(norb=norb, seed=7, multiplicity_per_sector=multiplicity,
        neighborhoods=state.neighborhoods, bond_multiplets=state.bond_dimensions,
        bond_magnetic_dimensions=state.magnetic_bond_dimensions,
        assembly_seconds=assembly_seconds, compilation_seconds=compilation_seconds,
        native_build_seconds=native_build_seconds, component_build_seconds=component_build_seconds,
        native_complete_action_seconds=native_seconds,
        component_complete_action_seconds=component_seconds,
        component_kernel_only_seconds=component_kernel_seconds,
        complete_action_speed_ratio=component_seconds/native_seconds,
        kernel_only_speed_ratio=component_kernel_seconds/native_seconds,
        native_environment_bytes=native_environment_bytes,
        component_environment_bytes=component_environment_bytes,
        environment_storage_ratio=component_environment_bytes/native_environment_bytes,
        native_mpo_bytes=sum(a.nbytes for s in mpo.sites for a in s.data.values()),
        component_mpo_bytes=sum(a.nbytes for a in components),
        local_action_max_absolute_error=absolute_error,
        note='Equivalent-state microbenchmark; excludes sweep caching, convergence, and Abelian/DMRG comparisons.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--norb', type=int, default=4)
    parser.add_argument('--multiplicity', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = benchmark(args.norb, args.multiplicity, args.repeats)
    report = json.dumps(result, indent=2)
    print(report)
    if args.output:
        args.output.write_text(report+'\n')
