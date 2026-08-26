"""A compact two-site MPS DMRG baseline for snake-ordered spin lattices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

from ..operators import LatticeMPO


def _extend_left(environment, tensor, mpo_tensor):
    return np.einsum(
        "aij,ixk,abxy,jyl->bkl",
        environment,
        tensor.conj(),
        mpo_tensor,
        tensor,
        optimize=True,
    )


def _extend_right(environment, tensor, mpo_tensor):
    return np.einsum(
        "abxy,ixk,jyl,bkl->aij",
        mpo_tensor,
        tensor.conj(),
        tensor,
        environment,
        optimize=True,
    )


class SnakeMPS:
    """Open-boundary MPS with tensors ordered as ``(left, physical, right)``."""

    def __init__(self, tensors):
        tensors = [np.asarray(tensor).copy() for tensor in tensors]
        if not tensors:
            raise ValueError("an MPS must contain at least one tensor.")
        physical_dim = tensors[0].shape[1] if tensors[0].ndim == 3 else None
        for site, tensor in enumerate(tensors):
            if tensor.ndim != 3:
                raise ValueError("MPS tensors must have three axes.")
            if tensor.shape[1] != physical_dim:
                raise ValueError("MPS physical dimensions must be uniform.")
            if site == 0 and tensor.shape[0] != 1:
                raise ValueError("the first MPS left bond must have dimension one.")
            if site == len(tensors) - 1 and tensor.shape[-1] != 1:
                raise ValueError("the final MPS right bond must have dimension one.")
            if site and tensors[site - 1].shape[-1] != tensor.shape[0]:
                raise ValueError(f"MPS virtual bond mismatch before site {site}.")
            if not np.all(np.isfinite(tensor)):
                raise ValueError("MPS tensors must contain finite values.")
        self.tensors = tensors
        self.physical_dim = int(physical_dim)

    @classmethod
    def random(
        cls,
        nsites,
        *,
        physical_dim=2,
        bond_dim=8,
        seed=None,
        real=True,
    ):
        nsites = int(nsites)
        physical_dim = int(physical_dim)
        bond_dim = int(bond_dim)
        if nsites <= 0 or physical_dim <= 0 or bond_dim <= 0:
            raise ValueError("MPS dimensions must be positive.")
        rng = np.random.default_rng(seed)
        tensors = []
        for site in range(nsites):
            left_dim = 1 if site == 0 else bond_dim
            right_dim = 1 if site == nsites - 1 else bond_dim
            tensor = rng.normal(size=(left_dim, physical_dim, right_dim))
            if not real:
                tensor = tensor + 1j * rng.normal(size=tensor.shape)
            tensor /= np.sqrt(tensor.size)
            tensors.append(tensor)
        state = cls(tensors)
        state.right_canonicalize()
        state.normalize()
        return state

    @property
    def nsites(self):
        return len(self.tensors)

    @property
    def bond_dimensions(self):
        return tuple(tensor.shape[-1] for tensor in self.tensors[:-1])

    def copy(self):
        return SnakeMPS([tensor.copy() for tensor in self.tensors])

    def state_vector(self, *, max_sites=20):
        if self.nsites > max_sites:
            raise ValueError("refusing to materialize an exponentially large state.")
        wavefunction = self.tensors[0][0]
        for tensor in self.tensors[1:]:
            wavefunction = np.tensordot(
                wavefunction,
                tensor,
                axes=([-1], [0]),
            )
        return wavefunction[..., 0].reshape(-1)

    def norm(self):
        environment = np.ones((1, 1), dtype=np.result_type(*self.tensors))
        for tensor in self.tensors:
            environment = np.einsum(
                "ij,isk,jsl->kl",
                environment,
                tensor.conj(),
                tensor,
                optimize=True,
            )
        return float(np.real(environment[0, 0]))

    def normalize(self):
        norm = self.norm()
        if norm <= np.finfo(float).tiny:
            raise ValueError("cannot normalize a zero MPS.")
        self.tensors[0] /= np.sqrt(norm)
        return self

    def right_canonicalize(self):
        for site in range(self.nsites - 1, 0, -1):
            tensor = self.tensors[site]
            matrix = tensor.reshape(tensor.shape[0], -1)
            q_matrix, r_matrix = np.linalg.qr(matrix.T, mode="reduced")
            self.tensors[site] = q_matrix.T.reshape(
                q_matrix.shape[1],
                tensor.shape[1],
                tensor.shape[2],
            )
            self.tensors[site - 1] = np.tensordot(
                self.tensors[site - 1],
                r_matrix.T,
                axes=([-1], [0]),
            )
        return self

    def expectation(self, mpo):
        if not isinstance(mpo, LatticeMPO):
            raise TypeError("mpo must be a LatticeMPO.")
        if mpo.nsites != self.nsites or mpo.physical_dim != self.physical_dim:
            raise ValueError("MPS and MPO dimensions do not match.")
        environment = np.ones(
            (1, 1, 1),
            dtype=np.result_type(*self.tensors, *mpo.factors),
        )
        for tensor, mpo_tensor in zip(self.tensors, mpo.factors):
            environment = _extend_left(environment, tensor, mpo_tensor)
        numerator = environment[0, 0, 0]
        return float(np.real(numerator / self.norm()))


@dataclass(frozen=True)
class MPSDMRGOptions:
    max_sweeps: int = 6
    tolerance: float = 1.0e-9
    eigensolver_tolerance: float = 1.0e-10
    eigensolver_max_iterations: int = 300
    verbosity: int = 0


@dataclass(frozen=True)
class MPSSweep:
    sweep: int
    energy: float
    energy_change: float
    energy_density_change: float
    max_discarded_weight: float


@dataclass(frozen=True)
class MPSDMRGResult:
    state: SnakeMPS
    energy: float
    converged: bool
    sweeps: int
    history: tuple[MPSSweep, ...]
    message: str


def _build_left_environments(state, mpo):
    environments = [None] * (state.nsites + 1)
    environments[0] = np.ones((1, 1, 1), dtype=np.result_type(*state.tensors))
    for site in range(state.nsites):
        environments[site + 1] = _extend_left(
            environments[site],
            state.tensors[site],
            mpo.factors[site],
        )
    return environments


def _build_right_environments(state, mpo):
    environments = [None] * (state.nsites + 1)
    environments[-1] = np.ones((1, 1, 1), dtype=np.result_type(*state.tensors))
    for site in range(state.nsites - 1, -1, -1):
        environments[site] = _extend_right(
            environments[site + 1],
            state.tensors[site],
            mpo.factors[site],
        )
    return environments


def _compatible_transition_pairs(first_transitions, second_transitions):
    return tuple(
        (left_channel, right_channel, first_operator, second_operator)
        for left_channel, middle, first_operator in first_transitions
        for second_middle, right_channel, second_operator in second_transitions
        if middle == second_middle
    )


def _two_site_action(left, right, transition_pairs, shape, vector):
    theta = np.asarray(vector).reshape(shape)
    result = np.zeros(shape, dtype=np.result_type(left, right, theta))
    for left_channel, right_channel, first_operator, second_operator in (
        transition_pairs
    ):
        value = np.tensordot(
            left[left_channel],
            theta,
            axes=([1], [0]),
        )
        value = np.tensordot(
            first_operator,
            value,
            axes=([1], [1]),
        ).transpose(1, 0, 2, 3)
        value = np.tensordot(
            second_operator,
            value,
            axes=([1], [2]),
        ).transpose(1, 2, 0, 3)
        value = np.tensordot(
            value,
            right[right_channel],
            axes=([3], [1]),
        )
        result += value
    return result.reshape(-1)


def _lowest_two_site_vector(
    left,
    right,
    first_transitions,
    second_transitions,
    theta,
    options,
):
    shape = theta.shape
    dimension = theta.size
    transition_pairs = _compatible_transition_pairs(
        first_transitions,
        second_transitions,
    )

    def action(vector):
        return _two_site_action(
            left,
            right,
            transition_pairs,
            shape,
            vector,
        )

    if dimension <= 64:
        identity = np.eye(dimension, dtype=np.result_type(theta))
        matrix = np.column_stack(
            [action(identity[:, column]) for column in range(dimension)]
        )
        matrix = 0.5 * (matrix + matrix.conj().T)
        values, vectors = linalg.eigh(
            matrix,
            subset_by_index=[0, 0],
            check_finite=False,
        )
        return float(np.real(values[0])), vectors[:, 0]

    operator = LinearOperator(
        (dimension, dimension),
        matvec=action,
        dtype=np.result_type(left, right, theta),
    )
    initial = theta.reshape(-1)
    initial /= np.linalg.norm(initial)
    try:
        values, vectors = eigsh(
            operator,
            k=1,
            which="SA",
            v0=initial,
            tol=options.eigensolver_tolerance,
            maxiter=options.eigensolver_max_iterations,
        )
    except ArpackNoConvergence as error:
        if error.eigenvectors is None or error.eigenvectors.shape[1] == 0:
            raise
        values = error.eigenvalues
        vectors = error.eigenvectors
    return float(np.real(values[0])), vectors[:, 0]


def _split_theta(theta, bond_dim, direction):
    left_dim, first_physical, second_physical, right_dim = theta.shape
    matrix = theta.reshape(
        left_dim * first_physical,
        second_physical * right_dim,
    )
    left_vectors, singular_values, right_vectors = np.linalg.svd(
        matrix,
        full_matrices=False,
    )
    keep = min(int(bond_dim), singular_values.size)
    total_weight = float(np.sum(singular_values**2))
    discarded_weight = float(np.sum(singular_values[keep:] ** 2))
    if total_weight:
        discarded_weight /= total_weight
    left_vectors = left_vectors[:, :keep]
    singular_values = singular_values[:keep]
    right_vectors = right_vectors[:keep]
    singular_values /= np.linalg.norm(singular_values)
    if direction == "lr":
        first = left_vectors.reshape(left_dim, first_physical, keep)
        second = (singular_values[:, None] * right_vectors).reshape(
            keep,
            second_physical,
            right_dim,
        )
    elif direction == "rl":
        first = (left_vectors * singular_values[None, :]).reshape(
            left_dim,
            first_physical,
            keep,
        )
        second = right_vectors.reshape(keep, second_physical, right_dim)
    else:
        raise ValueError("direction must be 'lr' or 'rl'.")
    return first, second, discarded_weight


def _optimize_bond(state, mpo, site, left, right, bond_dim, direction, options):
    theta = np.tensordot(
        state.tensors[site],
        state.tensors[site + 1],
        axes=([-1], [0]),
    )
    _energy, vector = _lowest_two_site_vector(
        left,
        right,
        mpo.transitions[site],
        mpo.transitions[site + 1],
        theta,
        options,
    )
    optimized = vector.reshape(theta.shape)
    first, second, discarded_weight = _split_theta(
        optimized,
        bond_dim,
        direction,
    )
    state.tensors[site] = first
    state.tensors[site + 1] = second
    return discarded_weight


def mps_dmrg(
    hamiltonian,
    *,
    state=None,
    bond_dim=8,
    seed=None,
    options=None,
):
    """Run two-site finite DMRG on the snake-ordered Hamiltonian."""

    if not isinstance(hamiltonian, LatticeMPO):
        raise TypeError("hamiltonian must be a LatticeMPO.")
    options = MPSDMRGOptions() if options is None else options
    if not isinstance(options, MPSDMRGOptions):
        raise TypeError("options must be an MPSDMRGOptions instance.")
    if options.max_sweeps <= 0 or options.tolerance <= 0.0:
        raise ValueError("DMRG sweep controls must be positive.")
    if bond_dim <= 0:
        raise ValueError("bond_dim must be positive.")
    if state is None:
        state = SnakeMPS.random(
            hamiltonian.nsites,
            physical_dim=hamiltonian.physical_dim,
            bond_dim=bond_dim,
            seed=seed,
        )
    elif not isinstance(state, SnakeMPS):
        raise TypeError("state must be a SnakeMPS.")
    else:
        state = state.copy()
        state.right_canonicalize().normalize()
    if state.nsites != hamiltonian.nsites:
        raise ValueError("MPS and Hamiltonian lengths do not match.")

    previous_energy = state.expectation(hamiltonian)
    history = []
    converged = False
    message = "STOP: MAXIMUM SWEEPS REACHED"
    for sweep in range(1, options.max_sweeps + 1):
        discarded_weights = []
        right_environments = _build_right_environments(state, hamiltonian)
        left = np.ones((1, 1, 1), dtype=np.result_type(*state.tensors))
        for site in range(state.nsites - 1):
            discarded_weights.append(
                _optimize_bond(
                    state,
                    hamiltonian,
                    site,
                    left,
                    right_environments[site + 2],
                    bond_dim,
                    "lr",
                    options,
                )
            )
            left = _extend_left(
                left,
                state.tensors[site],
                hamiltonian.factors[site],
            )

        left_environments = _build_left_environments(state, hamiltonian)
        right = np.ones((1, 1, 1), dtype=np.result_type(*state.tensors))
        for site in range(state.nsites - 2, -1, -1):
            discarded_weights.append(
                _optimize_bond(
                    state,
                    hamiltonian,
                    site,
                    left_environments[site],
                    right,
                    bond_dim,
                    "rl",
                    options,
                )
            )
            right = _extend_right(
                right,
                state.tensors[site + 1],
                hamiltonian.factors[site + 1],
            )

        energy = state.expectation(hamiltonian)
        energy_change = abs(energy - previous_energy)
        energy_density_change = energy_change / state.nsites
        history.append(
            MPSSweep(
                sweep=sweep,
                energy=energy,
                energy_change=energy_change,
                energy_density_change=energy_density_change,
                max_discarded_weight=max(discarded_weights, default=0.0),
            )
        )
        if options.verbosity:
            print(
                f"snake MPS sweep {sweep:3d}  energy={energy:.14f}  "
                f"dE/site={energy_density_change:.3e}  "
                f"discarded={history[-1].max_discarded_weight:.3e}"
            )
        if energy_density_change <= options.tolerance:
            converged = True
            message = "CONVERGENCE: SWEEP ENERGY DENSITY CHANGE <= TOLERANCE"
            break
        previous_energy = energy

    return MPSDMRGResult(
        state=state,
        energy=history[-1].energy,
        converged=converged,
        sweeps=len(history),
        history=tuple(history),
        message=message,
    )
