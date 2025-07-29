import numpy as np
import os
import h5py
import sys
import matplotlib.pyplot as plt
from pyqed.floquet.utils import track_valence_band, berry_phase_winding, figure, track_valence_band_GL2013, save_data_to_hdf5, load_data_from_hdf5
from  pyqed.floquet.floquet import TightBinding, FloquetBloch
import h5py
import gc 

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
                
                term_E = dE_dt * np.vdot(phi_m, dHF_dE @ phi_n)
                term_O = dO_dt * np.vdot(phi_m, dHF_dO @ phi_n)
                
                off_diagonal_term += (term_E + term_O) / denom * C[n]

            # Combine the terms to get the full time derivative for C_m
            dC_dt_vec[m] = dynamical_term - off_diagonal_term # MODIFIED: A_{mn} has 1/(en-em) so total is 1/(em-en) = -1/(en-em)
            
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
        
        # MODIFIED: Array to store NAC magnitude
        nac_magnitude_vs_time = np.zeros(self.num_time_steps, dtype=float)
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
            
            # MODIFIED: Calculate NAC magnitude |<m|d/dE|n>| for m=0, n=1
            if self.norbs >= 2:
                phi_m = states_current[0] # Corresponds to band 0
                phi_n = states_current[1] # Corresponds to band 1
                numerator = np.vdot(phi_m, dHF_dE @ phi_n)
                print(numerator)
                np.set_printoptions(threshold=sys.maxsize)
                # print(f"phi_m: \n {phi_m[78:84]}, \n phi_n: \n {phi_n[78:84]}, \n dHF_dE: \n {dHF_dE[78:84,78:84]}, \n numerator: {numerator}")
                # print(f"phi_m: \n {phi_m}, \n phi_n: \n {phi_n}, \ndHF_dE: \n {dHF_dE}, \n numerator: \n{numerator}")
                # print(f"phi_m: \n {phi_m.shape}, \n phi_n: \n {phi_n}, \ndHF_dE: \n {dHF_dE[8:14,8:14]}, \n numerator: \n{numerator}")
                denominator = energies_current[1] - energies_current[0]
                # if abs(denominator) > 1e-9:
                nac_magnitude_vs_time[t_idx] = np.abs(numerator / denominator)
                nac_magnitude_vs_time[t_idx] = np.abs(numerator)
                # else:
                #     nac_magnitude_vs_time[t_idx] = 0.0 # Assign 0 if states are degenerate
            
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
            
        # MODIFIED: Return both populations and NAC magnitudes
        return C_t, nac_magnitude_vs_time

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

def save_evolution_data(filename, C_t, nac_magnitudes):
    """Saves time evolution data to an HDF5 file."""
    with h5py.File(filename, 'w') as f:
        f.create_dataset('C_t', data=C_t)
        f.create_dataset('nac_magnitudes', data=nac_magnitudes)

def load_evolution_data(filename):
    """Loads time evolution data from an HDF5 file."""
    with h5py.File(filename, 'r') as f:
        C_t = f['C_t'][:]
        nac_magnitudes = f['nac_magnitudes'][:]
    return C_t, nac_magnitudes


if __name__ == "__main__":
    # np.set_printoptions(precision=4, linewidth=180) # self_use, print nicely for checking the matrix elements
    from scipy.special import jv, jvp
    # --- Simulation Parameters ---
    coords = [[0], [0.75]]

    # # --- realistic parameters setup ---
    # tb_model = TightBinding(coords, lambda_decay=1.0, lattice_constant=[1.0], relative_Hopping=[0.1286,0.0919])
    # total_time = 500
    # num_time_steps = 10000
    # omega = 0.1
    # nt_assigned = 81
    # t_space = np.linspace(0, total_time, num_time_steps + 1)
    # E_max = 2
    # # --- Setup output and cache directories ---
    # output_plot_dir = 'MacBook_local_data/realistic_parameter/dynamics_plots'
    # evolution_cache_dir = 'MacBook_local_data/realistic_parameter/evolution_cache'
    # os.makedirs(output_plot_dir, exist_ok=True)
    # os.makedirs(evolution_cache_dir, exist_ok=True)

    # ---simplified parameters for test purpose---
    tb_model = TightBinding(coords, lambda_decay=1.0, lattice_constant=[1.0], relative_Hopping=[1.5,1])
    total_time = 50
    total_time = 2*np.pi+np.pi/10
    # num_time_steps = 10000
    num_time_steps = 5000
    omega = 10
    nt_assigned = 11
    # --- Setup output and cache directories ---
    output_plot_dir = 'MacBook_local_data/model_parameter/dynamics_plots'
    evolution_cache_dir = 'MacBook_local_data/model_parameter/evolution_cache'
    os.makedirs(output_plot_dir, exist_ok=True)
    os.makedirs(evolution_cache_dir, exist_ok=True)
    t_space = np.linspace(-total_time/2, total_time/2, num_time_steps+1)
    E_max = 200


    # --- Gaussian pulse parameters ---
    t0 = 0
    sigma = total_time / 10.0
    epsilon_path = E_max * np.exp(-((t_space - t0)**2) / (2 * sigma**2))
    omega_path = np.full_like(t_space, 10.0)

    # --- K-space scan setup ---
    num_k_points = 101
    k_grid = np.linspace(-np.pi, np.pi, num_k_points)
    k_grid = np.linspace(-np.pi, 0, int(num_k_points//2)+1) #only calculate half for fast testing, corrected later 
    final_excited_populations = []
    


    # --- Main Loop over k-points ---
    for k_val in k_grid:
        print(f"--- Processing k = {k_val:.4f} ---")
        
        # Define a unique filename for the evolution cache based on key parameters
        cache_filename = os.path.join(
            evolution_cache_dir,
            f"k{k_val:.4f}_Emax{E_max:.3f}_T{total_time:.3f}_sigma{sigma:.3f}.h5"
        )

        # Check for the final evolution data first
        if os.path.exists(cache_filename):
            print(f"Loading cached evolution data from {cache_filename}")
            C_t, nac_magnitudes = load_evolution_data(cache_filename)
        else:
            # If no cached data, run the full pre-computation and evolution
            print("No cache found. Running full calculation...")
            
            # --- Step 1: Run Floquet pre-computation ---
            data_path_k = f'MacBook_local_data/model_parameter/data_k_scan/k_{k_val:.4f}_Emax{E_max:.3f}_T{total_time:.3f}_sigma{sigma:.3f}'
            floquet_model = tb_model.Floquet(
                omegad=omega, E0=np.unique(epsilon_path), nt=nt_assigned,
                polarization=[1], data_path=data_path_k
            )
            floquet_model.run(k=[k_val])

            # --- Step 2: Run the time evolution ---
            time_evolver = TimeEvolution(
                floquet_model=floquet_model, total_time=total_time,
                num_time_steps=num_time_steps, initial_state_band_index=0
            )
            C_t, nac_magnitudes = time_evolver.run_evolution(epsilon_path, omega_path)
            
            # --- Step 3: Save the newly calculated data ---
            save_evolution_data(cache_filename, C_t, nac_magnitudes)
            print(f"Saved new evolution data to {cache_filename}")

        final_population_excited_state = np.abs(C_t[-1, 1])**2
        final_excited_populations.append(final_population_excited_state)
        print(f"Final excited population: {final_population_excited_state:.4f}")

        # --- Plotting and saving the individual dynamics figure ---
        plot_filename = os.path.join(output_plot_dir, 
                                     f"dyn_k{k_val:.4f}_Emax{E_max:.2f}_s{sigma:.2f}.png")
        
        # Optional: Also skip plotting if the plot already exists
        if os.path.exists(plot_filename):
             print(f"Plot already exists at {plot_filename}. Skipping.")
        else:
            fig, (ax1, ax3) = plt.subplots(
                nrows=2, ncols=1, figsize=(12, 10), sharex=True,
                gridspec_kw={'height_ratios': [2, 1]}
            )
            fig.suptitle(f'Time Evolution (k={k_val:.4f}, E_max={E_max:.1f}, $\sigma$={sigma:.2f})', 
                         fontsize=16, y=0.97)

            # (Your detailed plotting code...)
            color_pop = 'tab:blue'
            ax1.set_ylabel('Population $|C_n(t)|^2$', color=color_pop)
            for i in range(2): # Assuming norbs=2
                ax1.plot(t_space, np.abs(C_t[:, i])**2, label=f'State {i}')
            ax1.plot(t_space, np.sum(np.abs(C_t)**2, axis=1), label='Total', linestyle='--')
            ax1.tick_params(axis='y', labelcolor=color_pop)
            ax1.legend(loc='upper left')
            ax2 = ax1.twinx()
            # ax1.set_yscale('log')
            color_field = 'green'
            ax2.set_ylabel('E-Field', color=color_field)
            ax2.plot(t_space, epsilon_path, color=color_field, linestyle=':')
            ax2.tick_params(axis='y', labelcolor=color_field)
            
            color_nac = 'tab:red'
            ax3.set_ylabel('NAC Magnitude', color=color_nac)
            ax3.plot(t_space[:-1], nac_magnitudes, color=color_nac)
            ax3.tick_params(axis='y', labelcolor=color_nac)
            ax3.set_yscale('log')

            ax3.set_xlabel('Time (t)')

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(plot_filename)
            plt.close(fig)
            print(f"Saved dynamics plot to {plot_filename}")
        # forces free up memory from the large objects created in this iteration (floquet_model, C_t, figures, etc.).
        gc.collect()
    # --- Final Summary Plot (after the loop) ---
    plt.figure(figsize=(10, 6))
    plt.plot(k_grid, final_excited_populations, marker='o', markersize=2, linestyle='-')
    plt.xlabel('Crystal Momentum (k)')
    plt.ylabel('Final Excitation Population')
    plt.title(f'Final Population vs. k (E_max={E_max:.1f}, $\sigma$={sigma:.2f})')
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$\pi$',r'$\pi/2$','0', r'$\pi/2$', r'$\pi$'])
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.show()