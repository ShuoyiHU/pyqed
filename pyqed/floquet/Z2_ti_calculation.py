import numpy as np
import sys
from scipy import linalg
from scipy.special import jv
from pyqed.mol import Mol, dag
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# from tqdm import tqdm
import time
import os
from pyqed.floquet.utils import track_valence_band, berry_phase_winding, figure, track_valence_band_GL2013, save_data_to_hdf5, load_data_from_hdf5
from numpy import exp, eye, zeros, arctan2
from scipy.linalg import eigh
import os
import h5py
import math
import matplotlib.colors as colors
import gc 

class Z2TICalculation:
    """
    Class for calculating the Z2 topological invariant for a time-reversal invariant (TRI) system.
    This class implements the method described in Fu & Kane, PRB 74, 195312 (2006).
    """
    def __init__(self, data_path, E0_scalar=None, norbs=2, nt=1):
        """
        Initializes the Z2TICalculation instance.

        Args:
            data_path (str): Path to the directory containing the band structure data files.
            E0_scalar (float, optional): The scalar field amplitude for the TRI Hamiltonian.
                                         If None, it will be set to 0.0.
            norbs (int): Total number of orbitals in the system (must be even).
            nt (int): Number of time slices in the Floquet-Bloch basis.
        """
        self.data_path = data_path
        self.E0_scalar = E0_scalar if E0_scalar is not None else 0.0
        self.norbs = norbs
        self.nt = nt
        self.k = None
        self._E_list = None  # List of field amplitudes for which data is available

    def _winding_number(self, num_occupied_bands, E=None):
            # test of new method calculating the topological invariant based on Time reversal polarization and a Z 2 adiabatic spin pump, 10.1103/PhysRevB.74.195312
            """
            Calculates the Z2 invariant (P_theta mod 2) for a single time-reversal
            invariant (TRI) Hamiltonian. This is suitable for classifying a static system.

            Args:
                E_val (float): The field amplitude of the TRI Hamiltonian (typically E=0).
                num_occupied_bands (int): The number of occupied bands. Must be an even number.

            Returns:
                int: The Z2 invariant, 0 (trivial) or 1 (topological).
            """
            if self.k is None:
                raise RuntimeError("Must call .run() to define a k-path before this calculation.")
            if num_occupied_bands % 2 != 0:
                raise ValueError("Number of occupied bands must be even for Z2 invariant calculation.")
            if self.norbs % 2 != 0:
                raise ValueError("The total number of orbitals must be even (spinful system).")
            # 3) build list of E values
            if E is None:
                E_list = self._E_list if self._E_list is not None else [self.E0_scalar]
            elif isinstance(E, (float, int)):
                E_list = [float(E)]
            else:
                E_list = list(E)

            z2_invariant = []

            # 4) loop over each field amplitude
            for E_val in E_list:
                # --- Load data for the specified E_val ---
                fname = os.path.join(self.data_path, f"band_E{E_val:.6f}.h5")
                print(f"Loading data for E = {E_val:.6f} from {fname}...")
                
                try:
                    _, all_states = load_data_from_hdf5(fname)
                except FileNotFoundError:
                    print(f"Error: Data file not found for E={E_val:.6f}. Cannot calculate Z2 invariant.")
                    return np.nan

                # The loaded `all_states` is a list of arrays. We need the first `num_occupied_bands`.
                occupied_states = np.array(all_states[:num_occupied_bands])
                
                # Reshape from (num_bands, Nk, NF) to (Nk, num_bands, NF) for the helper function
                occupied_states = np.transpose(occupied_states, (1, 0, 2))
                
                # --- Calculate the P_theta invariant ---
                print("Calculating P_theta (Pfaffian winding)...")
                P_theta = self._calculate_P_theta(occupied_states)
                
                # The Z2 invariant is P_theta mod 2
                z2_invariant.append(int(abs(P_theta)) % 2)
                
            return z2_invariant    
    @staticmethod
    def _pfaffian(A):
        """
        Computes the Pfaffian of a complex skew-symmetric matrix A.
        The algorithm is based on a block LU decomposition with pivoting.
        """
        n_rows, n_cols = A.shape
        if n_rows != n_cols or n_rows % 2 != 0:
            raise ValueError("Matrix for Pfaffian must be square with even dimensions.")
        
        if not np.allclose(A, -A.T, atol=1e-9):
            raise ValueError("Matrix must be skew-symmetric.")

        A_copy = A.copy().astype(np.complex128)
        pfaff_val = 1.0
        n = n_rows // 2

        for k in range(n):
            # Pivoting: find the largest element in the remaining submatrix
            sub_matrix = A_copy[2*k:, 2*k:]
            pivot_pos = np.unravel_index(np.argmax(np.abs(sub_matrix)), sub_matrix.shape)
            pivot_row, pivot_col = pivot_pos[0] + 2*k, pivot_pos[1] + 2*k

            # Swap rows/columns to bring pivot to the (2*k, 2*k+1) position
            if pivot_row != 2*k:
                A_copy[[2*k, pivot_row], :] = A_copy[[pivot_row, 2*k], :]
                A_copy[:, [2*k, pivot_row]] = A_copy[:, [pivot_row, 2*k]]
                pfaff_val *= -1.0
            
            if pivot_col != 2*k + 1:
                A_copy[[2*k+1, pivot_col], :] = A_copy[[pivot_col, 2*k+1], :]
                A_copy[:, [2*k+1, pivot_col]] = A_copy[:, [pivot_col, 2*k+1]]
                pfaff_val *= -1.0

            # Elimination
            pivot_val = A_copy[2*k, 2*k+1]
            if abs(pivot_val) < 1e-12: return 0.0

            pfaff_val *= pivot_val
            inv_pivot = 1.0 / pivot_val
            
            i_range = np.arange(2*k + 2, n_rows)
            if len(i_range) > 0:
                v = A_copy[2*k, i_range]
                w = A_copy[2*k+1, i_range]
                update_matrix = np.outer(v, w) - np.outer(w, v)
                A_copy[np.ix_(i_range, i_range)] += inv_pivot * update_matrix

        return pfaff_val

    def _construct_z2_theta_operator(self):
        """
        Constructs the matrix for the unitary part of the time-reversal operator (i*sigma_y)
        in the full Floquet-Bloch basis.
        """
        # Operator in the orbital basis (assumes spin-1/2 basis: up, down)
        i_sigma_y_orb = np.array([[0, 1], [-1, 0]], dtype=complex)
        
        # Extend to the norbs-dimensional orbital space
        num_orbital_blocks = self.norbs // 2
        theta_orb_space = np.kron(np.eye(num_orbital_blocks, dtype=int), i_sigma_y_orb)
        
        # Extend to the full Floquet space (nt * norbs)
        theta_floquet_space = np.kron(np.eye(self.nt, dtype=int), theta_orb_space)
        
        return theta_floquet_space

    def _calculate_P_theta(self, occupied_states):
        """
        Calculates the P_theta invariant (winding number of the Pfaffian)
        for a single time-reversal invariant Hamiltonian.

        Args:
            occupied_states (np.ndarray): Array of shape (Nk, num_occupied, NF)
                                          containing the eigenvectors.

        Returns:
            float: The integer winding number of the Pfaffian phase.
        """
        theta_op = self._construct_z2_theta_operator()
        pfaffian_path = []
        
        for i in range(occupied_states.shape[0]): # Loop over k-points
            # U_k is a matrix of occupied eigenvectors: shape (NF, num_occupied)
            U_k = occupied_states[i, :, :].T
            
            # m(k) = U_k^† * (iσ_y) * U_k^*
            m_k = U_k.conj().T @ theta_op @ U_k.conj()
            
            pfaffian_path.append(self._pfaffian(m_k))

        # Calculate winding number from the list of complex Pfaffian values
        phases = np.angle(np.array(pfaffian_path))
        unwrapped_phases = np.unwrap(phases)
        winding_number = (unwrapped_phases[-1] - unwrapped_phases[0]) / (2 * np.pi)

        return np.round(winding_number)

    def calculate_z2_pump_invariant(self, E_t0, E_t_half, num_occupied_bands):
        """
        Calculates the Z2 invariant for an adiabatic pump cycle.

        This method implements the formula Δ = (P_theta(t=T/2) - P_theta(t=0)) mod 2,
        as described in Fu & Kane, PRB 74, 195312 (2006).

        Args:
            E_t0 (float): The field amplitude at the first time-reversal invariant (TRI) point.
            E_t_half (float): The field amplitude at the second TRI point.
            num_occupied_bands (int): The number of occupied bands to include in the calculation.
                                      Must be an even number.

        Returns:
            int: The Z2 invariant, 0 (trivial) or 1 (topological).
        """
        if self.k is None:
            raise RuntimeError("Must call .run() to define a k-path before this calculation.")
        if num_occupied_bands % 2 != 0:
            raise ValueError("Number of occupied bands must be even.")

        # --- Load data for the first TRI point (t=0) ---
        fname_t0 = os.path.join(self.data_path, f"band_E{E_t0:.6f}.h5")
        print(f"Loading data for E = {E_t0:.6f} from {fname_t0}...")
        _, all_states_t0 = load_data_from_hdf5(fname_t0)
        occupied_states_t0 = np.array(all_states_t0[:num_occupied_bands])
        # Reshape from (num_bands, Nk, NF) to (Nk, num_bands, NF)
        occupied_states_t0 = np.transpose(occupied_states_t0, (1, 0, 2))
        
        # --- Load data for the second TRI point (t=T/2) ---
        fname_t_half = os.path.join(self.data_path, f"band_E{E_t_half:.6f}.h5")
        print(f"Loading data for E = {E_t_half:.6f} from {fname_t_half}...")
        _, all_states_t_half = load_data_from_hdf5(fname_t_half)
        occupied_states_t_half = np.array(all_states_t_half[:num_occupied_bands])
        occupied_states_t_half = np.transpose(occupied_states_t_half, (1, 0, 2))

        # --- Calculate P_theta for each point ---
        print("\nCalculating P_theta for the E_t0 Hamiltonian...")
        P_theta_0 = self._calculate_P_theta(occupied_states_t0)
        print(f"-> P_theta(t=0) winding number = {P_theta_0}")

        print("\nCalculating P_theta for the E_t_half Hamiltonian...")
        P_theta_half = self._calculate_P_theta(occupied_states_t_half)
        print(f"-> P_theta(t=T/2) winding number = {P_theta_half}")

        # --- Compute the final Z2 invariant ---
        z2_invariant = int(abs(P_theta_half - P_theta_0)) % 2
        
        print("\n" + "="*45)
        print(f"Z2 Pump Invariant Δ = (P_theta(T/2) - P_theta(0)) mod 2")
        print(f"Δ = ({P_theta_half} - {P_theta_0}) mod 2 = {z2_invariant}")
        print("="*45)

        return z2_invariant