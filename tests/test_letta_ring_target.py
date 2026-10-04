"""Covariant periodic target closure against independent small physical states."""
from collections import Counter

import numpy as np
import pytest

from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
from pyqed.mps.symmetry import Sector
from pyqed.mps.nonabelian.tensor import NonabelianTensor
from pyqed.mps.nonabelian.coupling import ordered_two_m_values
from pyqed._letta_one_site_opt import ReducedPhysicalBasis
from pyqed._letta_one_site_opt.reduced_symmetry import _fuse_sectors
from pyqed._letta_one_site_opt.reduced_ring_target import ReducedRingTarget, dual_sector, signed_sector, signed_physical_basis
from pyqed._letta_one_site_opt.reduced_ring_contraction import CyclicReducedOperator, CyclicReducedNorm
from test_letta_native_ring import explicit_ring_vector
from test_letta_qchem_symmetry import total_operators, dense_mpo
from test_letta_qchem import integrals, determinant_hamiltonian


def random_target_ring(n, basis, target, *, copies=2, seed=51, anchor_two_j=0):
    rng = np.random.default_rng(seed)
    vacuum = Sector(basis.sectors[0].labels, tuple(
        SU2Irrep(anchor_two_j) if label == 'su2' else 0
        for label in basis.sectors[0].labels))
    # Multiple copies on BOTH sides of the closure keep a real virtual loop.
    bonds = [{vacuum}]
    for i in range(n):
        bonds.append({qr for ql in bonds[-1] for qp in basis.sectors
                      for qr in _fuse_sectors(ql, qp)})
    dual = dual_sector(target)
    bonds[-1] = {q for q in bonds[-1] if vacuum in _fuse_sectors(q, dual)}
    for i in range(n-1, -1, -1):
        bonds[i] = {q for q in bonds[i] if any(qr in bonds[i+1]
                    for qp in basis.sectors for qr in _fuse_sectors(q, qp))}
    bonds = [tuple(q for q in sorted(layer) for _ in range(copies)) for layer in bonds]
    bases = [basis]*n + [ReducedPhysicalBasis(('dual',), (dual,), (1,))]
    sites = []
    for i, physical in enumerate(bases):
        left, right = bonds[i], bonds[i+1] if i < n else bonds[0]
        data = {}
        for ql, dl in Counter(left).items():
            for qp, dp in zip(physical.sectors, physical.multiplicities):
                for qr, dr in Counter(right).items():
                    if qr in _fuse_sectors(ql, qp):
                        shape = (dl, dp, dr)
                        data[ql, qp, qr] = (rng.normal(size=shape)+1j*rng.normal(size=shape))/3
        qns = [left, tuple(q for q, r in zip(physical.sectors, physical.multiplicities) for _ in range(r)), right]
        sites.append(NonabelianTensor(data=data, qns=qns, dirs=[-1, 1, 1],
                                     metadata={'physical_basis': 'fully_reduced_su2'}))
    return ReducedRingTarget(tuple(sites[:-1]), sites[-1], target)


@pytest.mark.parametrize('anchor_two_j', [0, 1])
@pytest.mark.parametrize('charge,two_s', [(1, 1), (2, 0), (2, 2), (3, 1)])
def test_covariant_ring_fixes_charge_and_spin_in_every_component(charge, two_s, anchor_two_j, monkeypatch):
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    import pyqed._letta_one_site_opt.reduced_contraction as magnetic
    n = 3
    basis = signed_physical_basis(ReducedPhysicalBasis.spatial_orbital())
    ring = random_target_ring(n, basis, SpinChargeSector(charge, SU2Irrep(two_s)), anchor_two_j=anchor_two_j)
    components = explicit_ring_vector(ring.sites).reshape(4**n, two_s+1)
    N, Z, S2 = total_operators(n)
    h, g = integrals(n)
    problem = ElectronicProblem(h, g, ((charge+two_s)//2, (charge-two_s)//2))
    operator = problem.su2_mpo().native_mpo(basis)
    # Independently built determinant operator, no optimizer internals.
    dense = determinant_hamiltonian(problem)
    norms = []
    energies = []
    for index, aux_m in enumerate(ordered_two_m_values(SU2Irrep(two_s))):
        v = components[:, index]
        np.testing.assert_allclose(N@v, charge*v, atol=2e-12)
        np.testing.assert_allclose(Z@v, -aux_m/2*v, atol=2e-12)
        np.testing.assert_allclose(S2@v, two_s/2*(two_s/2+1)*v, atol=2e-12)
        norms.append(np.vdot(v, v).real)
        energies.append(np.vdot(v, dense@v).real/norms[-1])
    np.testing.assert_allclose(norms, norms[0], atol=2e-12)
    np.testing.assert_allclose(energies, energies[0], atol=2e-11)
    def forbidden(*args, **kwargs):
        raise AssertionError('target ring contraction expanded variational magnetic tensors')
    monkeypatch.setattr(magnetic, 'expand_reduced_mps_site', forbidden)
    total_norm = ring.norm_squared()
    assert total_norm.real > 1e-12
    assert total_norm == pytest.approx(sum(norms), abs=2e-12)
    assert ring.component_norm_squared() == pytest.approx(norms[0], abs=2e-12)
    extended = ring.extend_hamiltonian(operator)
    value = CyclicReducedOperator(ring.sites, extended).overlap()
    assert value/total_norm == pytest.approx(energies[0], abs=2e-11)
    assert len(ring.sites[0].qns[0]) == len(ring.closure.qns[2]) == 2


def test_target_closure_local_hamiltonian_and_metric_actions():
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    from pyqed._letta_one_site_opt.reduced_frontier import _BlockVectorLayout
    basis = signed_physical_basis(ReducedPhysicalBasis.spatial_orbital())
    ring = random_target_ring(2, basis, SpinChargeSector(1, SU2Irrep(1)))
    problem = ElectronicProblem(*integrals(2), (1, 0))
    h = ring.extend_hamiltonian(problem.su2_mpo().native_mpo(basis))
    dense = np.kron(determinant_hamiltonian(problem), np.eye(2))
    for site in (0, len(ring.sites)-1):
        layout = _BlockVectorLayout({key: a.shape for key, a in ring.sites[site].data.items()})
        frame = []
        for x in np.eye(layout.size):
            trial = list(ring.sites)
            trial[site] = trial[site].copy()
            trial[site].data = layout.unpack(x)
            frame.append(explicit_ring_vector(trial))
        frame = np.column_stack(frame)
        for chain, reference in [(CyclicReducedNorm(ring.sites), frame.conj().T@frame),
                                  (CyclicReducedOperator(ring.sites, h), frame.conj().T@dense@frame)]:
            actual = np.column_stack([layout.pack(chain.local_action(site, layout.unpack(x)))
                                      for x in np.eye(layout.size)])
            np.testing.assert_allclose(actual, reference, atol=3e-12)


def test_dual_product_charges_and_invalid_closure_are_explicit():
    from pyqed.mps.symmetry import Sector
    q = Sector(('nalpha', 'nbeta', 'pg', 'su2'), (2, 1, 3, SU2Irrep(1)))
    assert dual_sector(q) == Sector(q.labels, (-2, -1, 3, SU2Irrep(1)))
    ring = random_target_ring(2, signed_physical_basis(ReducedPhysicalBasis.spatial_orbital()), SpinChargeSector(2, SU2Irrep(0)))
    with pytest.raises(ValueError, match='dual target'):
        ReducedRingTarget(ring.physical_sites, ring.closure, SpinChargeSector(1, SU2Irrep(1)))
    invalid = ring.closure.copy()
    key = next(iter(invalid.data))
    bad = signed_sector(SpinChargeSector(9, SU2Irrep(0)))
    invalid.data[key[0], key[1], bad] = invalid.data.pop(key)
    with pytest.raises(ValueError, match='invalid reduced ring block'):
        CyclicReducedNorm(ring.physical_sites + (invalid,))


@pytest.mark.parametrize('model', ['bose', 'fermion_two_charges'])
def test_native_abelian_target_ring_energy_and_sector(model):
    from pyqed._letta_one_site_opt.reduced_mpo_compile import SpinTensorMPO
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    n = 3
    if model == 'bose':
        labels = ('number', 'su2')
        sectors = tuple(Sector(labels, (i, SU2Irrep(0))) for i in range(3))
        target = Sector(labels, (3, SU2Irrep(0)))
        basis = ReducedPhysicalBasis(('0', '1', '2'), sectors, (1, 1, 1))
        annihilation = np.diag(np.sqrt([1., 2.]), 1)
        number = np.diag([0., 1., 2.])
        terms = []
        for i in range(n):
            onsite = [np.eye(3) for _ in range(n)]
            onsite[i] = 1.5*number@(number-np.eye(3))
            terms.append(onsite)
            j = (i+1) % n
            for left, right in [(annihilation.T, annihilation), (annihilation, annihilation.T)]:
                local = [np.eye(3) for _ in range(n)]
                local[i], local[j] = -left, right
                terms.append(local)
        k = len(terms)
        first = np.stack([term[0] for term in terms])[None]
        middle = np.zeros((k, k, 3, 3))
        for i, term in enumerate(terms):
            middle[i, i] = term[1]
        last = np.stack([term[2] for term in terms])[:, None]
        mpo = SpinTensorMPO.compile((first, middle, last), basis)
        dense = sum((np.kron(np.kron(t[0], t[1]), t[2]) for t in terms), np.zeros((27, 27)))
        charges = [(i,) for i in range(3)]
        expected_charge = (3,)
    else:
        labels = ('nalpha', 'nbeta', 'su2')
        charges = [(0, 0), (1, 0), (0, 1), (1, 1)]
        sectors = tuple(Sector(labels, q+(SU2Irrep(0),)) for q in charges)
        target = Sector(labels, (1, 1, SU2Irrep(0)))
        basis = ReducedPhysicalBasis(('empty', 'alpha', 'beta', 'double'), sectors, (1,)*4)
        problem = ElectronicProblem(*integrals(n), (1, 1))
        mpo = problem.su2_mpo().native_mpo(basis)
        dense = determinant_hamiltonian(problem)
        expected_charge = (1, 1)
    ring = random_target_ring(n, basis, target)
    vector = explicit_ring_vector(ring.sites)
    configurations = list(np.ndindex(*((basis.dense_dim,)*n)))
    for amplitude, configuration in zip(vector, configurations):
        charge = tuple(sum(charges[i][axis] for i in configuration) for axis in range(len(expected_charge)))
        if charge != expected_charge:
            assert abs(amplitude) < 1e-13
    expected = np.vdot(vector, dense@vector)
    actual = CyclicReducedOperator(ring.sites, ring.extend_hamiltonian(mpo)).overlap()
    assert actual == pytest.approx(expected, abs=4e-12)
    assert ring.norm_squared() == pytest.approx(np.vdot(vector, vector), abs=3e-12)
