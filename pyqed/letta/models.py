"""Model definitions independent of LETTA topology, dependencies and updates."""
from dataclasses import dataclass
from operator import index

import numpy as np

from .._letta_one_site_opt.abelian_backend import AbelianReducedMap
from .._letta_one_site_opt.operators import LatticeMPO
from .._letta_one_site_opt.qchem import ElectronicProblem
from .._letta_one_site_opt.reduced_operators import ReducedMPOHamiltonian
from .._letta_one_site_opt.reduced_symmetry import ReducedPhysicalBasis, ReducedSymmetry
from .._letta_one_site_opt.symmetry import AbelianSymmetry


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be an integer')
    try:
        result = index(value)
    except TypeError as error:
        raise ValueError(f'{name} must be an integer') from error
    if minimum is not None and result < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    return result


def _real(value, name):
    try:
        if np.ndim(value) or np.iscomplexobj(value) or not np.isfinite(value):
            raise ValueError
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be a finite real scalar') from error


def _edges(nsites, periodic, bonds):
    if not isinstance(periodic, (bool, np.bool_)):
        raise ValueError('periodic must be a boolean')
    if bonds is not None and periodic:
        raise ValueError('specify explicit bonds or a periodic chain, not both')
    if bonds is None:
        return tuple((i, i+1) for i in range(nsites-1))+(
            ((0, nsites-1),) if periodic and nsites > 2 else ())
    result = []
    for edge in bonds:
        if len(edge) != 2:
            raise ValueError('each bond must have two site indices')
        i, j = sorted(_integer(x, 'bond index') for x in edge)
        if i == j or j >= nsites or (i, j) in result:
            raise ValueError('bonds must be distinct, in range and without self edges')
        result.append((i, j))
    return tuple(result)


def _product_sum(nsites, dimension, terms):
    """Exact direct-sum product MPO; compilation subsequently reduces its bonds."""
    terms = tuple((coefficient, operators) for coefficient, operators in terms if coefficient != 0)
    if not terms:
        terms = ((0., {}),)
    identity = np.eye(dimension, dtype=complex)
    if nsites == 1:
        local = sum(c*ops.get(0, identity) for c, ops in terms)
        return LatticeMPO((local[None, None],))
    width = len(terms)
    first = np.zeros((1, width, dimension, dimension), dtype=complex)
    last = np.zeros((width, 1, dimension, dimension), dtype=complex)
    for k, (c, ops) in enumerate(terms):
        first[0, k] = c*ops.get(0, identity)
        last[k, 0] = ops.get(nsites-1, identity)
    factors = [first]
    for i in range(1, nsites-1):
        core = np.zeros((width, width, dimension, dimension), dtype=complex)
        for k, (_, ops) in enumerate(terms):
            core[k, k] = ops.get(i, identity)
        factors.append(core)
    return LatticeMPO(tuple(factors+[last]))


def _mpo_norm(factors):
    """Frobenius norm by exact local QR, without a many-body operator matrix."""
    boundary = np.ones((1, 1), dtype=complex)
    for core in factors:
        value = np.einsum('ab,bcst->acst', boundary, core, optimize=True)
        value = value.transpose(0, 2, 3, 1).reshape(-1, value.shape[1])
        _, boundary = np.linalg.qr(value, mode='reduced')
    return float(np.linalg.norm(boundary))


def _check_hermitian(native):
    # Build H-H† before taking its norm. Subtracting two scalar squared norms
    # would lose the very precision this validation is intended to check.
    factors = tuple(native.component_factors())
    difference = []
    for i, core in enumerate(factors):
        adjoint = core.conj().swapaxes(2, 3)
        if len(factors) == 1:
            difference.append(core-adjoint)
        elif i == 0:
            difference.append(np.concatenate((core, -adjoint), axis=1))
        elif i == len(factors)-1:
            difference.append(np.concatenate((core, adjoint), axis=0))
        else:
            left, right, d, _ = core.shape
            block = np.zeros((2*left, 2*right, d, d), dtype=complex)
            block[:left, :right] = core
            block[left:, right:] = adjoint
            difference.append(block)
    norm, error = _mpo_norm(factors), _mpo_norm(difference)
    if not np.isfinite(norm) or not np.isfinite(error) or error > 1e-10*max(norm, np.finfo(float).tiny):
        raise ValueError('Hamiltonian must be Hermitian in the physical inner product')


@dataclass(frozen=True)
class LETTAProblem:
    """Hamiltonian and target; virtual boundaries and ties belong to the state.

    Canonical MPO physical indices follow ``symmetry.physical_basis``. For an
    Abelian model, ``physical_order`` maps that order back to its original local
    basis. The molecular orbital order is determined by the supplied integrals.
    """
    hamiltonian: ReducedMPOHamiltonian
    symmetry: ReducedSymmetry
    name: str = 'custom LETTA problem'
    physical_order: tuple[int, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.hamiltonian, ReducedMPOHamiltonian):
            raise TypeError('hamiltonian must be a ReducedMPOHamiltonian')
        if not isinstance(self.symmetry, ReducedSymmetry):
            raise TypeError('symmetry must be a ReducedSymmetry')
        if self.hamiltonian.contraction_backend != 'reduced':
            raise ValueError('public symmetry optimization requires the native reduced backend')
        dimension = self.symmetry.physical_basis.dense_dim
        order = tuple(range(dimension)) if self.physical_order is None else tuple(self.physical_order)
        if len(order) != dimension or set(order) != set(range(dimension)):
            raise ValueError('physical_order must be a physical-basis permutation')
        object.__setattr__(self, 'physical_order', order)
        # Conservation and basis errors are input errors, never local recovery.
        _check_hermitian(self.hamiltonian.native_mpo(self.symmetry.physical_basis))
        self.symmetry.reachable_bond_sectors(self.nsites)

    @property
    def nsites(self):
        return len(self.hamiltonian)

    @property
    def physical_basis(self):
        return self.symmetry.physical_basis


def _abelian_problem(mpo, symmetry, name):
    mapping = AbelianReducedMap(symmetry)
    return LETTAProblem(mapping.hamiltonian(mpo), mapping.reduced_symmetry,
                        name, mapping.physical_order)


def molecular(problem, *, symmetry='su2', two_s=None):
    """Construct a model from real chemists' integrals in ElectronicProblem.

    Modes: su2 (N,S), n (N), n_sz (N,2Sz), nalpha_nbeta. The u1 spelling selects
    n_sz. For su2, two_s is twice the target total spin, not spin projection.
    """
    if not isinstance(problem, ElectronicProblem):
        raise TypeError('molecular requires an ElectronicProblem')
    mode = 'n_sz' if symmetry == 'u1' else symmetry
    if two_s is not None:
        two_s = _integer(two_s, 'two_s')
    target = problem.symmetry(mode, two_s=two_s)
    if mode == 'su2':
        return LETTAProblem(problem.su2_mpo(), target, 'molecular')
    return _abelian_problem(problem.mpo(backend='symbolic'), target, 'molecular')


def hubbard(nsites, *, nelec, t=1., U=4., mu=0., periodic=False, bonds=None,
            symmetry='su2', two_s=None):
    """Fermionic Hubbard on a chain or explicit graph of spatial orbitals.

    periodic affects Hamiltonian edges only. A two-site chain has one bond;
    duplicate bonds are not introduced by periodic=True.
    """
    n = _integer(nsites, 'nsites', 1)
    t, U, mu = (_real(v, k) for v, k in ((t, 't'), (U, 'U'), (mu, 'mu')))
    h = -mu*np.eye(n)
    for i, j in _edges(n, periodic, bonds):
        h[i, j] = h[j, i] = -t
    eri = np.zeros((n,)*4)
    for i in range(n):
        eri[i, i, i, i] = U
    model = molecular(ElectronicProblem(h, eri, nelec), symmetry=symmetry, two_s=two_s)
    return LETTAProblem(model.hamiltonian, model.symmetry, 'fermionic Hubbard', model.physical_order)


def bose_hubbard(nsites, *, particles, max_occupancy, t=1., U=4., mu=0.,
                 periodic=False, bonds=None):
    """Spinless Bose-Hubbard with exact U(1) number and a local occupation cutoff."""
    n = _integer(nsites, 'nsites', 1)
    cutoff = _integer(max_occupancy, 'max_occupancy', 1)
    particles = _integer(particles, 'particles')
    if particles > n*cutoff:
        raise ValueError('particle number exceeds the local occupation capacity')
    t, U, mu = (_real(v, k) for v, k in ((t, 't'), (U, 'U'), (mu, 'mu')))
    a = np.diag(np.sqrt(np.arange(1, cutoff+1)), 1)
    number = np.diag(np.arange(cutoff+1, dtype=float))
    onsite = U/2*number@(number-np.eye(cutoff+1))-mu*number
    terms = [(1., {i: onsite}) for i in range(n)]
    for i, j in _edges(n, periodic, bonds):
        terms.extend(((-t, {i: a.T, j: a}), (-t, {i: a, j: a.T})))
    return _abelian_problem(_product_sum(n, cutoff+1, terms),
        AbelianSymmetry(tuple(range(cutoff+1)), particles, name='U(1) N'), 'Bose-Hubbard')


def heisenberg(nsites, *, J=1., periodic=False, bonds=None, symmetry='su2',
               two_s=None, two_sz=None):
    """Spin-half Heisenberg, fixing total S or spin projection Sz.

    Canonical physical order is down, up. SU(2) invariant ties have only one
    physical multiplet label for this model and add no extra spin dependence.
    """
    n = _integer(nsites, 'nsites', 1)
    J = _real(J, 'J')
    sx = .5*np.array([[0., 1.], [1., 0.]])
    sy = .5*np.array([[0., 1j], [-1j, 0.]])
    sz = .5*np.diag([-1., 1.])
    terms = [(J, {i: op, j: op}) for i, j in _edges(n, periodic, bonds) for op in (sx, sy, sz)]
    mpo = _product_sum(n, 2, terms)
    if symmetry == 'su2':
        if two_sz is not None:
            raise ValueError('use two_s for SU(2), not two_sz')
        spin = n % 2 if two_s is None else _integer(two_s, 'two_s')
        target = ReducedSymmetry.su2(ReducedPhysicalBasis.spin_half(), target_two_j=spin)
        return LETTAProblem(ReducedMPOHamiltonian(None, mpo.factors), target, 'Heisenberg')
    if symmetry != 'u1' or two_s is not None:
        raise ValueError('Heisenberg symmetry must be su2 with two_s or u1 with two_sz')
    projection = n % 2 if two_sz is None else _integer(two_sz, 'two_sz', None)
    return _abelian_problem(mpo, AbelianSymmetry((-1, 1), projection, name='U(1) 2Sz'), 'Heisenberg')
