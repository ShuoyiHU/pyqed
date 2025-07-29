import numpy as np
import os
import h5py

import sys
from scipy import linalg
from scipy.special import jv
from pyqed.mol import Mol, dag
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# from tqdm import tqdm
import time
from pyqed.floquet.utils import track_valence_band, berry_phase_winding, figure, track_valence_band_GL2013, save_data_to_hdf5, load_data_from_hdf5
from  pyqed.floquet.floquet import TightBinding, FloquetBloch
from numpy import exp, eye, zeros, arctan2
from scipy.linalg import eigh
import h5py
import math

class TimeEvolution:
    """
    Handles the time evolution of a quantum state using pre-computed Floquet data.
    """
    def __init__(self, floquet_model, total_time, num_time_steps, initial_state_band_index=0):
        if floquet_model.k is None:
            raise ValueError("The `floquet_model` must be run before time evolution.")

        self.floquet_model = floquet_model
        self.total_time = total_time
        self.num_time_steps = num_time_steps
        self.dt = total_time / num_time_steps
        self.initial_band = initial_state_band_index
        self.norbs = floquet_model.norbs
        self.k_point = floquet_model.k[0] # Assumes evolution at a single k-point

    def _calculate_dC_dt(self, C, energies_current, states_current, dHF_dE, dHF_dO, dE_dt, dO_dt):
        """
        Calculates the time derivative of the state coefficients C.
        This is the f(C) part of the ODE, assuming parameters are constant over dt.
        """
        num_bands = self.norbs
        dC_dt_vec = np.zeros(num_bands, dtype=complex)

        for m in range(num_bands):
            # 1. Dynamical Phase Term
            dynamical_term = -1j * energies_current[m] * C[m]
            
            # 2. Off-Diagonal Coupling Term
            off_diagonal_term = 0
            for n in range(num_bands):
                if m == n:
                    continue

                denom = energies_current[m] - energies_current[n]
                if np.abs(denom) < 1e-9:
                    continue
                
                phi_m = states_current[m]
                phi_n = states_current[n]
                
                # Note: Corrected a typo from the original code in the term_O calculation.
                term_E = dE_dt * np.vdot(phi_m, dHF_dE @ phi_n)
                # print(f"term_E: {term_E}, dE_dt: {dE_dt}, phi_m: {phi_m}, dHF_dE: {dHF_dE}, phi_n: {phi_n}")
                term_O = dO_dt * np.vdot(phi_m, dHF_dO @ phi_n)
                
                off_diagonal_term += (term_E + term_O) / denom * C[n]

            # Combine the terms to get the full time derivative for C_m
            dC_dt_vec[m] = dynamical_term + off_diagonal_term
            
        return dC_dt_vec

    def run_evolution(self, epsilon_path, omega_path):
        """
        Calculates the evolution of the state coefficients C_n(t) using the RK4 method.
        """
        num_bands = self.norbs
        C = np.zeros(num_bands, dtype=complex)
        C[self.initial_band] = 1.0

        C_t = np.zeros((self.num_time_steps + 1, num_bands), dtype=complex)
        C_t[0] = C
        
        # Cache for loaded Floquet data to improve performance
        data_cache = {}

        for t_idx in range(self.num_time_steps):
            E0_current = epsilon_path[t_idx]
            Omega_current = omega_path[t_idx]
            cache_key = f"{E0_current:.6f}"
            
            # Populate cache if data for the current E0 is not present
            if cache_key not in data_cache:
                energies, states = self.load_floquet_data(E0_current)
                dHF_dE = self.floquet_model.build_derivative_H(self.k_point, E0_current, Omega_current, 'epsilon')
                dHF_dO = self.floquet_model.build_derivative_H(self.k_point, E0_current, Omega_current, 'omega')
                data_cache[cache_key] = (energies, states, dHF_dE, dHF_dO)
                
            energies_current, states_current, dHF_dE, dHF_dO = data_cache[cache_key]
            
            # Calculate derivatives of parameters for this time step
            dE_dt = (epsilon_path[t_idx+1] - E0_current) / self.dt if t_idx < self.num_time_steps else 0
            dO_dt = (omega_path[t_idx+1] - Omega_current) / self.dt if t_idx < self.num_time_steps else 0

            # Define a lambda function for f(C) for conciseness in RK4 steps
            f_C = lambda C_vec: self._calculate_dC_dt(C_vec, energies_current, states_current, dHF_dE, dHF_dO, dE_dt, dO_dt)

            # RK4 steps
            k1 = f_C(C)
            k2 = f_C(C + 0.5 * self.dt * k1)
            k3 = f_C(C + 0.5 * self.dt * k2)
            k4 = f_C(C + self.dt * k3)

            # Update C using the RK4 formula
            C += (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            
            # Re-normalize to counteract numerical drift
            # C = C / np.linalg.norm(C)

            C_t[t_idx + 1] = C
            
        return C_t

    def load_floquet_data(self, E0):
        """ Helper to load energies and states for a specific E0. """
        fname = os.path.join(self.floquet_model.data_path, f"band_E{E0:.6f}.h5")
        try:
            energies_k, states_k_list = load_data_from_hdf5(fname)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found for E0={E0:.6f} at {fname}. Please ensure floquet_model.run() was executed for all E values in epsilon_path.")
        # Assuming we evolve at the first k-point and need all bands
        energies = energies_k[0, :]
        states = [s[0, :] for s in states_k_list]
        return energies, states


if __name__ == "__main__":
    # Precondition setup
    coords = [[0], [0.75]]
    tb_model = TightBinding(coords, lambda_decay=1.0, lattice_constant=[1.0], relative_Hopping=[1.5, 1.0])

    total_time = 8000
    num_time_steps = 110000
    t_space = np.linspace(0, total_time, num_time_steps + 1)
    E_max = 200
    a = int(num_time_steps // 11)
    # Construct the path for the electric field amplitude
    epsilon_path_1 = np.linspace(0, E_max, 5*a+1)
    epsilon_path_2 = np.full(a, E_max)  
    epsilon_path_3 = np.linspace(E_max, 0, 5*a+1)
    epsilon_path = np.concatenate((epsilon_path_1, epsilon_path_2, epsilon_path_3)) 

    # Construct the path for the driving frequency
    omega_path_1 = np.full(5*a+1, 10.0) 
    omega_path_2 = np.linspace(10.0, 10.0, a) # Corrected to have same start and end
    omega_path_3 = np.full(5*a+1, 10.0) 
    omega_path = np.concatenate((omega_path_1, omega_path_2, omega_path_3)) 

    # Ensure paths have the correct length for time evolution loop
    # The loop needs epsilon_path[t_idx+1], so path must be num_time_steps + 1 long.
    # The constructed path is 11002 long, which is fine.
    
    k_point_for_evolution = [np.pi-0.2]
    
    # Instantiate and run FloquetBloch to pre-compute eigenstates
    floquet_model = tb_model.Floquet(
        omegad=10.0,
        E0=np.unique(epsilon_path),
        nt=21,
        polarization=[1],
        data_path=f'./MacBook_local_data/floquet_data_evolution_fd/k={k_point_for_evolution[0]:.2f}',
    )

    # This step pre-computes and saves the Floquet states needed for the evolution
    floquet_model.run(k=k_point_for_evolution)

    # Instantiate TimeEvolution with the pre-computed Floquet model
    time_evolver = TimeEvolution(
        floquet_model=floquet_model,
        total_time=total_time,
        num_time_steps=num_time_steps,
        initial_state_band_index=1 # Starting in the upper band
    )

    # Run the time evolution using the new RK4 method
    C_t = time_evolver.run_evolution(epsilon_path, omega_path)

    # PLOT: State Populations
    plt.figure(figsize=(8, 6))
    total_population = np.zeros_like(t_space, dtype=float)
    for i in range(floquet_model.norbs):
        plt.plot(t_space, np.abs(C_t[:, i])**2, label=f'State {i} Population')
        total_population += np.abs(C_t[:, i])**2
    
    # Plot total population to check for conservation (should be close to 1)
    plt.plot(t_space, total_population, label='Total Population', linestyle='--', color='black')
    
    plt.xlabel('Time (t)')
    plt.ylabel('Population $|C_n(t)|^2$')
    plt.title('Time Evolution of Floquet State Populations (RK4)')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.05, 1.05) # Set y-axis limits for better visualization
    plt.show()