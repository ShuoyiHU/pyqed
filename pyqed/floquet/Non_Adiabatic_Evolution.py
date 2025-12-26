#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 18:53:56 2018

@author: binggu
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import linalg
from scipy.special import jv
from pyqed.mol import Mol, dag
# from tqdm import tqdm
import time
from pyqed.floquet.utils import track_valence_band, berry_phase_winding, figure, track_valence_band_GL2013, save_data_to_hdf5, load_data_from_hdf5
from  pyqed.floquet.floquet import TightBinding, FloquetBloch
from numpy import exp, eye, zeros, arctan2
from scipy.linalg import eigh
import os
import h5py
import math
from scipy.integrate import solve_ivp

if __name__ == '__main__':
    """
    Main execution block to set up and run a quantum dynamics simulation.
    The electric field amplitude E0 follows a "round-trip" path, and the basis
    is constructed from Floquet modes at multiple points along this path.
    Includes functionality to save and load simulation results to avoid re-calculation.
    """

    # =========================================================================
    #   STEP 1: DEFINE THE SYSTEM AND ESTABLISH THE EXPANDED FLOQUET BASIS
    # =========================================================================
    print("--- STEP 1: Setting up the system and calculating the expanded basis ---")
    
    # --- System Parameters ---
    omega = 10.0
    E_start = 0.1
    E_peak = 1.0
    nt = 5
    k_point = [-np.pi]
    b = 0.7
    
    # --- Parameter for the expanded basis ---
    nE_basis = 5 # Number of E-field points to build the basis from
    
    # --- Model Instantiation ---
    coords = [[0.0], [b]]
    tb_model = TightBinding(coords, relative_Hopping=[1.5, 1.0])
    floquet_system_basis = tb_model.Floquet(
        omegad=omega, 
        E0=E_start, # E0 here is just a placeholder for the class
        nt=nt, 
        polarization=[1.0],
        data_path=f'MacBook_local_data/temp_data/b_{b:.2f}/'
    )
    
    # --- Create the grid of E-field values for the basis ---
    E_basis_set = np.linspace(E_start, E_peak, nE_basis)
    print(f"Constructing basis from {nE_basis} E-field points: {np.round(E_basis_set, 2)}")

    # --- Loop through the E-field grid to build the full basis ---
    all_quasi_energies = []
    all_floquet_evecs = []
    
    _, states_at_zero = floquet_system_basis.track_band(k_values=k_point, E0=0)
    previous_states = states_at_zero

    for E_val in E_basis_set:
        print(f"  Calculating modes for E0 = {E_val:.2f}...")
        energies, evecs_list = floquet_system_basis.track_band(
            k_values=k_point, 
            E0=E_val,
            previous_state=previous_states
        )
        
        if not evecs_list or evecs_list[0].size == 0:
            print(f"Error: Could not calculate basis at E0 = {E_val}. Exiting.")
            sys.exit()
            
        all_quasi_energies.extend(energies[0])
        all_floquet_evecs.extend(evecs_list)
        previous_states = evecs_list

    print("\nFull basis constructed successfully.")
    E_diag_matrix = np.diag(all_quasi_energies)
    num_modes = len(all_quasi_energies)
    print(f"Total number of basis states: {num_modes}")

    nt_basis = floquet_system_basis.nt
    norbs = floquet_system_basis.norbs
    basis_fcfm = np.zeros((num_modes, nt_basis, norbs), dtype=complex)
    for i in range(num_modes):
        basis_fcfm[i, :, :] = all_floquet_evecs[i][0].reshape((nt_basis, norbs))

    # =========================================================================
    #   STEP 2: DEFINE THE DYNAMICS (PATH AND INITIAL STATE)
    # =========================================================================
    print("\n--- STEP 2: Defining the dynamics path and initial conditions ---")

    T_ramp_up = 0.1
    T_ramp_down = 0.1
    T_pulse = T_ramp_up + T_ramp_down
    
    def E_path(t):
        if t <= T_ramp_up:
            return E_start + (E_peak - E_start) * (t / T_ramp_up)
        elif t <= T_pulse:
            t_relative = t - T_ramp_up
            return E_peak - (E_peak - E_start) * (t_relative / T_ramp_down)
        else:
            return E_start
            
    C0 = np.zeros(num_modes, dtype=complex)
    C0[0] = 1.0 
    
    print(f"System will follow a round-trip path: E0 = {E_start} -> {E_peak} -> {E_start}")
    
    # =========================================================================
    #   STEP 3: BUILD AND RUN THE EOM SOLVER 
    # =========================================================================
    print("\n--- STEP 3: Building and running the EOM solver ---")

    # --- Define a unique filename for the results cache ---
    results_dir = "MacBook_local_data/simulation_results"
    os.makedirs(results_dir, exist_ok=True)
    results_filename = os.path.join(results_dir, f"dynamics_nE{nE_basis}_Epeak{E_peak:.1f}_Tramp{T_ramp_up:.1f}.h5")
    
    solution = None

    if os.path.exists(results_filename):
        print(f"Loading previously calculated dynamics data from: {results_filename}")
        with h5py.File(results_filename, 'r') as f:
            # Reconstruct a simple object that mimics the `solve_ivp` result
            from types import SimpleNamespace
            solution = SimpleNamespace()
            solution.t = f['t'][:]
            solution.y = f['y'][:]
            solution.status = 0 # Assume success if file exists
    else:
        print(f"No cached data found. Starting numerical integration...")
        E_basis_set_for_coupling = E_basis_set

        def dynamics_eom_rhs(t, C, model, E_diag, basis_fcfm_arg, k_pt):
            S_t = model.overlap_matrix_of_Floquet_modes(basis_fcfm_arg, t)
            M_t = model.Coupling_matrix(basis_fcfm_arg, k_pt, E_path(t), E_basis_set_for_coupling, t)
            
            try:
                S_inv_t = linalg.inv(S_t)
            except linalg.LinAlgError:
                print(f"Singular matrix S(t) at time t={t}. Halting.")
                return np.zeros_like(C)
                
            H_eff_term = (S_t @ E_diag) + M_t
            hbar = 1.0
            dC_dt = (1 / (1j * hbar)) * (S_inv_t @ H_eff_term @ C)
            print(f"Time elapsed for t={t}: {time.time() - start_time:.2f} seconds")
            return dC_dt

        t_final = 0.3
        t_span = (0, t_final)
        t_eval = np.linspace(t_span[0], t_final, 101)
        
        solver_args = (floquet_system_basis, E_diag_matrix, basis_fcfm, k_point)
        
        start_time = time.time()
        solution = solve_ivp(
            fun=dynamics_eom_rhs,
            t_span=t_span,
            y0=C0,
            t_eval=t_eval,
            args=solver_args,
            method='RK45',
        )
        end_time = time.time()
        print(f"Integration complete. Time elapsed: {end_time - start_time:.2f} seconds.")

        # --- Save the results if the integration was successful ---
        if solution.status == 0:
            print(f"Saving dynamics data to: {results_filename}")
            with h5py.File(results_filename, 'w') as f:
                f.create_dataset('t', data=solution.t)
                f.create_dataset('y', data=solution.y)

    # =========================================================================
    #   STEP 4: VISUALIZE THE RESULTS
    # =========================================================================
    if solution is not None and solution.status == 0:
        print("\n--- STEP 4: Visualizing the results ---")
        populations = np.abs(solution.y)**2
        
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        color = 'tab:blue'
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Population', fontsize=12, color=color)
        
        ax1.plot(solution.t, populations[0, :], label=f'Pop. of State 0 (from E₀={E_basis_set[0]:.1f})')
        ax1.plot(solution.t, populations[1, :], label=f'Pop. of State 1 (from E₀={E_basis_set[0]:.1f})')
        
        if num_modes > 2:
            other_pops = np.sum(populations[2:, :], axis=0)
            ax1.plot(solution.t, other_pops, label='Pop. of all other basis states', linestyle=':')

        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('E-Field Amplitude E₀(t)', fontsize=12, color=color)
        ax2.plot(solution.t, [E_path(t) for t in solution.t], color=color, linestyle='--', label='E₀(t) Path')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        fig.suptitle('Floquet State Population Dynamics During E-Field Pulse', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    else:
        print("\nCould not visualize results due to solver failure or no data.")

