# floquet_utils.py

import numpy as np
import h5py
import os
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.special import jv

def save_data_to_hdf5(filename, band_energy, band_eigenstates):
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Add this line
    with h5py.File(filename, 'w') as f:
        f.create_dataset('band_energy', data= band_energy)
        f.create_dataset('band_eigenstates', data = band_eigenstates)

def load_data_from_hdf5(filename):
    with h5py.File(filename, 'r') as f:
        return (f['band_energy'][:],
                f['band_eigenstates'][:])

def track_valence_band(k_values, T, E0, omega,
                       previous_val=None, previous_con=None,
                       v=0.15, w=0.2, nt=61, filename=None):
    from pyqed import Mol  # import locally to avoid circular import

    if filename and os.path.exists(filename):
        print(f"Loading data from {filename}...")
        return (*load_data_from_hdf5(filename), False)

    occ = np.zeros((2 * nt, len(k_values)), dtype=complex)
    con = np.zeros_like(occ)
    occ_e = np.zeros(len(k_values))
    con_e = np.zeros(len(k_values))

    for i, k0 in enumerate(k_values):
        mol = Mol(H0(k0, v, w), H1(k0))
        floquet = mol.Floquet(omegad=omega, E0=E0, nt=nt)
        if E0 == 0:
            quasi_occ, quasi_con = get_static_energies(H0(k0, v, w), w, k0)
            occ_state, occ_e[i] = floquet.winding_number_Peierls(T, k0, quasi_E=quasi_occ, w=w)
            con_state, con_e[i] = floquet.winding_number_Peierls(T, k0, quasi_E=quasi_con, w=w)
        else:
            occ_state, occ_e[i] = floquet.winding_number_Peierls(T, k0, previous_state=previous_val[:, i], w=w)
            con_state, con_e[i] = floquet.winding_number_Peierls(T, k0, previous_state=previous_con[:, i], w=w)
            if occ_e[i] > con_e[i]:
                occ_state, con_state = con_state, occ_state
                occ_e[i], con_e[i] = con_e[i], occ_e[i]
        occ[:, i], con[:, i] = occ_state, con_state

    if filename:
        save_data_to_hdf5(filename, occ, occ_e, con, con_e)

    return occ, occ_e, con, con_e, True

def berry_phase_winding(k_values, occ_states, nt=61):
    N = len(k_values)
    occ_states[:, 0] /= np.linalg.norm(occ_states[:, 0])
    proj = np.outer(occ_states[:, 0], np.conj(occ_states[:, 0]))
    for i in range(1, N):
        psi = occ_states[:, i] / np.linalg.norm(occ_states[:, i])
        proj = np.dot(proj, np.outer(psi, np.conj(psi)))
    angle = np.round(np.angle(np.trace(proj)),5)
    winding_number = (angle % (2 * np.pi)) / np.pi
    print(f"Winding number: {winding_number} \n")    
    return winding_number

def H0(k, v, w):
    return np.array([[0, v], [v, 0]], dtype=complex)

def H1(k):
    return np.array([[0, np.exp(-1j * k)], [np.exp(1j * k), 0]], dtype=complex)

def get_static_energies(H_static, w, k):
    H_eff = H_static + np.array([[0, w * np.exp(-1j * k)], [w * np.exp(1j * k), 0]], dtype=complex)
    eigvals, _ = linalg.eig(H_eff)
    eigvals = np.sort(eigvals.real)
    return eigvals[0], eigvals[1]

def figure(occ_state_energy, con_state_energy, k_values, E0, omega, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(8, 6))  # Wider aspect ratio (was 8x6)
    plt.plot(k_values, occ_state_energy, label=f'occ_state_E0 = {E0}, wavelength = {30/4.13/omega:.2f} nm')
    plt.plot(k_values, con_state_energy, label=f'con_state_E0 = {E0}, wavelength = {30/4.13/omega:.2f} nm')
    plt.xlabel(r'$k$ values')
    plt.ylabel(r'Quasienergies')
    plt.title(f'Floquet Band Structure for E₀ = {E0:.5g}, ω = {omega:.5g} (a.u.)')
    plt.legend()
    plt.grid(True)
    filename = f"{save_folder}/band_E0_{E0:.5f}_omega_{omega:.5f}.png"
    plt.savefig(filename, dpi=300)
    plt.close()


def track_valence_band_GL2013(k_values, E0_over_omega, previous_val=None, previous_con=None,
                             nt=61, filename=None, b=0.5, t=1.5, omega=100):
    from pyqed import Mol  # local import

    if filename and os.path.exists(filename):
        print(f"Loading data from {filename}...")
        return (*load_data_from_hdf5(filename), False)

    E0 = E0_over_omega * omega
    occ = np.zeros((2 * nt, len(k_values)), dtype=complex)
    con = np.zeros_like(occ)
    occ_e = np.zeros(len(k_values))
    con_e = np.zeros(len(k_values))

    for i, k0 in enumerate(k_values):
        H0 = np.array([[0, t * jv(0, E0_over_omega * b) + np.exp(-1j * k0) * jv(0, E0_over_omega * (1 - b))],
                       [t * jv(0, E0_over_omega * b) + np.exp(1j * k0) * jv(0, E0_over_omega * (1 - b)), 0]], dtype=complex)
        mol = Mol(H0, H1(k0))
        floquet = mol.Floquet(omegad=omega, E0=E0, nt=nt)

        if E0_over_omega == 0:
            eigvals = np.linalg.eigvalsh(H0)
            eigvals.sort()
            quasiE_val, quasiE_con = eigvals
            occ_state, occ_e[i] = floquet.winding_number_Peierls_GL2013(k0, quasi_E=quasiE_val, t=t, b=b, E_over_omega=E0_over_omega)
            con_state, con_e[i] = floquet.winding_number_Peierls_GL2013(k0, quasi_E=quasiE_con, t=t, b=b, E_over_omega=E0_over_omega)
        else:
            occ_state, occ_e[i] = floquet.winding_number_Peierls_GL2013(k0, previous_state=previous_val[:, i], t=t, b=b, E_over_omega=E0_over_omega)
            con_state, con_e[i] = floquet.winding_number_Peierls_GL2013(k0, previous_state=previous_con[:, i], t=t, b=b, E_over_omega=E0_over_omega)
            if occ_e[i] > con_e[i]:
                occ_state, con_state = con_state, occ_state
                occ_e[i], con_e[i] = con_e[i], occ_e[i]
        occ[:, i] = occ_state
        con[:, i] = con_state

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save_data_to_hdf5(filename, occ, occ_e, con, con_e)

    return occ, occ_e, con, con_e, True


def pfaffian(A):
    """
    Computes the Pfaffian of a complex skew-symmetric matrix A.

    The algorithm is based on a block LU decomposition with pivoting, which is
    numerically stable. The matrix A must be a 2n x 2n skew-symmetric matrix.

    Args:
        A (np.ndarray): A 2n x 2n skew-symmetric matrix.

    Returns:
        complex: The Pfaffian of the matrix A.
    """
    n_rows, n_cols = A.shape
    if n_rows != n_cols or n_rows % 2 != 0:
        raise ValueError("Matrix must be square and have even dimensions.")
    
    # Use a tolerance for the skew-symmetric check due to potential floating point errors
    if not np.allclose(A, -A.T, atol=1e-9):
        raise ValueError("Matrix must be skew-symmetric.")

    A_copy = A.copy().astype(np.complex128)
    pfaff_val = 1.0
    
    n = n_rows // 2

    for k in range(n):
        # --- Pivoting Step ---
        # Find the element with the largest absolute value in the submatrix
        # starting from (2*k, 2*k) to use as the pivot.
        sub_matrix = A_copy[2*k:, 2*k:]
        pivot_pos = np.unravel_index(np.argmax(np.abs(sub_matrix)), sub_matrix.shape)
        pivot_row, pivot_col = pivot_pos[0] + 2*k, pivot_pos[1] + 2*k

        # Bring the pivot to the (2*k, 2*k+1) position
        if pivot_row != 2*k:
            A_copy[[2*k, pivot_row], :] = A_copy[[pivot_row, 2*k], :]
            A_copy[:, [2*k, pivot_row]] = A_copy[:, [pivot_row, 2*k]]
            pfaff_val *= -1.0
        
        if pivot_col != 2*k + 1:
            A_copy[[2*k+1, pivot_col], :] = A_copy[[pivot_col, 2*k+1], :]
            A_copy[:, [2*k+1, pivot_col]] = A_copy[:, [pivot_col, 2*k+1]]
            pfaff_val *= -1.0

        # --- Elimination Step ---
        pivot_val = A_copy[2*k, 2*k+1]
        
        if abs(pivot_val) < 1e-12:
            return 0.0  # The matrix is singular, Pfaffian is 0.

        pfaff_val *= pivot_val
        inv_pivot = 1.0 / pivot_val
        
        # Update the remaining submatrix
        i_range = np.arange(2*k + 2, n_rows)
        if len(i_range) > 0:
            # Vectorized update for efficiency
            v = A_copy[2*k, i_range]
            w = A_copy[2*k+1, i_range]
            update_matrix = np.outer(v, w) - np.outer(w, v)
            A_copy[np.ix_(i_range, i_range)] += inv_pivot * update_matrix

    return pfaff_val


class Z2InvariantCalculator:
    """
    Calculates the Z2 topological invariant for a 1D adiabatic pump.

    This class implements the method described in Fu and Kane, Phys. Rev. B 74, 195312 (2006).
    The core of the calculation is based on Eq. 3.25, which defines the Z2 invariant `Δ` as
    the change in the "time-reversal polarization" `P_θ` between the two time-reversal
    invariant points of the pumping cycle.

    The time-reversal polarization `P_θ` is calculated using the formulation from
    Appendix A (Eq. A16), which expresses it as the winding number of the Pfaffian of
    the matrix m(k) = <u_i(k)|Θ|u_j(k)>, where |u(k)> are the occupied Bloch states
    and Θ is the time-reversal operator.
    """
    def __init__(self, basis_dim, num_occupied_bands):
        """
        Initializes the calculator.

        Args:
            basis_dim (int): The dimension of the system's basis (e.g., number of orbitals * 2 for spin).
                             This must be an even number.
            num_occupied_bands (int): The number of occupied bands. For a time-reversal
                                      invariant insulator, this must be an even number.
        """
        if basis_dim % 2 != 0:
            raise ValueError("Basis dimension must be even for a spinful system.")
        if num_occupied_bands % 2 != 0:
            raise ValueError("Number of occupied bands must be even for a TRI insulator.")

        self.basis_dim = basis_dim
        self.num_occupied_bands = num_occupied_bands
        self._theta_op = self._construct_time_reversal_operator()

    def _construct_time_reversal_operator(self):
        """
        Constructs the matrix representation of the time-reversal operator Θ = iσ_y K.
        We only need the unitary part, iσ_y, as the complex conjugation K is applied separately.
        
        This assumes a standard basis where spin-up and spin-down for each orbital are adjacent,
        e.g., (orb1_up, orb1_down, orb2_up, orb2_down, ...).
        """
        i_sigma_y = np.array([[0, 1], [-1, 0]], dtype=complex)
        num_orbital_blocks = self.basis_dim // 2
        # Create a block-diagonal matrix with i*sigma_y for each orbital block
        theta_matrix = np.kron(np.eye(num_orbital_blocks, dtype=int), i_sigma_y)
        return theta_matrix

    def _calculate_P_theta(self, occupied_states, k_path):
        """
        Calculates the integer-valued P_theta invariant for a single time-reversal
        invariant Hamiltonian. This corresponds to the winding number of the Pfaffian.

        Args:
            occupied_states (np.ndarray): Array of shape (Nk, num_occupied_bands, basis_dim) 
                                          containing the eigenvectors of the occupied bands.
            k_path (np.ndarray): Array of shape (Nk,) of k-points forming a closed loop
                                 in the Brillouin zone (e.g., from -pi to pi).

        Returns:
            float: The winding number of the Pfaffian phase, which should be an integer.
        """
        pfaffian_path = []
        for i in range(len(k_path)):
            # Get the matrix of occupied eigenvectors at k: shape (basis_dim, num_occupied_bands)
            U_k = occupied_states[i, :, :].T
            
            # Construct the matrix m(k) = U_k^† * (iσ_y) * U_k^*
            # This is equivalent to m_ij(k) = <u_i(k)|Θ|u_j(k)>
            m_k = U_k.conj().T @ self._theta_op @ U_k.conj()
            
            pfaffian_path.append(pfaffian(m_k))

        pfaffian_path = np.array(pfaffian_path)

        # Calculate the winding number of the complex path pfaffian_path
        phases = np.angle(pfaffian_path)
        unwrapped_phases = np.unwrap(phases)
        winding_number = (unwrapped_phases[-1] - unwrapped_phases[0]) / (2 * np.pi)

        return np.round(winding_number)

    def calculate_z2_pump_invariant(self, states_t0, states_t_half, k_path):
        """
        Calculates the Z2 invariant for an adiabatic pumping cycle using Eq. 3.25.

        This requires the eigenstates at the two time-reversal invariant points
        of the cycle, typically denoted t=0 and t=T/2.

        Args:
            states_t0 (np.ndarray): Occupied eigenvectors at the t=0 TRI point.
                                    Shape: (Nk, num_occupied_bands, basis_dim).
            states_t_half (np.ndarray): Occupied eigenvectors at the t=T/2 TRI point.
                                        Shape: (Nk, num_occupied_bands, basis_dim).
            k_path (np.ndarray): The path in the Brillouin zone, e.g., np.linspace(-pi, pi, num_k_points).

        Returns:
            int: The Z2 invariant, which will be 0 (trivial) or 1 (topological).
        """
        print("Calculating P_theta for the t=0 Hamiltonian...")
        P_theta_0 = self._calculate_P_theta(states_t0, k_path)
        print(f"-> P_theta(0) winding number = {P_theta_0}")

        print("\nCalculating P_theta for the t=T/2 Hamiltonian...")
        P_theta_half = self._calculate_P_theta(states_t_half, k_path)
        print(f"-> P_theta(T/2) winding number = {P_theta_half}")

        # The Z2 invariant is the difference in the P_theta invariants modulo 2
        z2_invariant = int(abs(P_theta_half - P_theta_0)) % 2
        
        print("\n" + "="*40)
        print(f"Z2 Pump Invariant Δ = (P_theta(T/2) - P_theta(0)) mod 2")
        print(f"Δ = ({P_theta_half} - {P_theta_0}) mod 2 = {z2_invariant}")
        print("="*40)

        return z2_invariant

# --- Example Usage ---
if __name__ == '__main__':
    # This is a placeholder for your data.
    # You will need to load your actual quantum state data here.
    
    # System parameters
    NK = 101  # Number of k-points
    BASIS_DIM = 4  # e.g., 2 orbitals, 2 spins
    NUM_OCCUPIED = 2 # Number of occupied bands (must be even)

    # 1. Define the k-path for the Brillouin Zone
    # It should be a closed path, e.g., from -pi to pi.
    k_path = np.linspace(-np.pi, np.pi, NK)

    # 2. Load or generate your state vectors.
    # The states should be numpy arrays of shape (NK, NUM_OCCUPIED, BASIS_DIM)
    # Here we generate random orthonormal states as a placeholder.
    # In your case, you would load these from your HDF5 files.
    print("Generating placeholder random states for demonstration...")
    
    def generate_random_occupied_states(nk, num_occ, basis_dim):
        states = np.zeros((nk, num_occ, basis_dim), dtype=complex)
        for i in range(nk):
            # Generate a random matrix and use QR decomposition to get orthonormal vectors
            random_matrix = np.random.rand(basis_dim, basis_dim) + 1j * np.random.rand(basis_dim, basis_dim)
            q, _ = np.linalg.qr(random_matrix)
            states[i, :, :] = q[:, :num_occ].T
        return states

    # Placeholder states for the two TRI points of the cycle
    mock_states_t0 = generate_random_occupied_states(NK, NUM_OCCUPIED, BASIS_DIM)
    mock_states_t_half = generate_random_occupied_states(NK, NUM_OCCUPIED, BASIS_DIM)
    
    print("Placeholder data generated.\n")

    # 3. Instantiate the calculator and run the calculation
    calculator = Z2InvariantCalculator(basis_dim=BASIS_DIM, num_occupied_bands=NUM_OCCUPIED)
    
    # Calculate the Z2 invariant for the pump
    z2_inv = calculator.calculate_z2_pump_invariant(mock_states_t0, mock_states_t_half, k_path)

    if z2_inv == 1:
        print("\nResult: The system is a non-trivial Z2 pump (topological).")
    else:
        print("\nResult: The system is a trivial Z2 pump.")

