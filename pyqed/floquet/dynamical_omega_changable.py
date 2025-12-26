import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
# Make sure the file containing your classes is accessible
from pyqed.floquet.floquet import TightBinding, FloquetBloch
from pyqed.floquet.utils import save_data_to_hdf5, load_data_from_hdf5
import gc

class TimeEvolution:
    """
    Handles the time evolution of a quantum state in the phase (phi) domain
    using pre-computed Floquet data for a chirped pulse.
    """
    def __init__(self, floquet_model, num_phi_steps, initial_state_band_index=0):
        self.floquet_model = floquet_model
        self.num_phi_steps = num_phi_steps
        self.initial_band = initial_state_band_index
        self.norbs = floquet_model.norbs
        self.k_point = floquet_model.k[0]

    def _calculate_dC_dphi(self, C, energies_current, states_current, omega_current, dHF_dF, dHF_dO, dF_dphi, dO_dphi):
        """
        Calculates the derivative of coefficients C with respect to phase phi.
        Based on the equation: dC_m/dphi = -i*(E_m/omega)*C_m - sum_{n!=m} <m|d/dphi|n>*C_n
        """
        num_bands = self.norbs
        dC_dphi_vec = np.zeros(num_bands, dtype=complex)

        for m in range(num_bands):
            # 1. Dynamical Phase Term
            dynamical_term = -1j * energies_current[m] * C[m] /omega_current
            
            # 2. Off-Diagonal Coupling Term
            off_diagonal_term = 0
            for n in range(num_bands):
                if m == n: 
                    continue
                
                # Denominator for Hellmann-Feynman theorem is (E_n - E_m)
                denom = energies_current[m] - energies_current[n]
                if np.abs(denom) < 1e-9: 
                    continue
                
                phi_m = states_current[m]
                phi_n = states_current[n]
                
                # Numerator of the coupling: <m|dH/dphi|n>
                vdot_F = np.vdot(phi_m, dHF_dF @ phi_n)
                vdot_O = np.vdot(phi_m, dHF_dO @ phi_n)
                coupling_numerator = dF_dphi * vdot_F + dO_dphi * vdot_O
                
                # Full coupling <m|d/dphi|n>
                coupling = coupling_numerator / denom
                
                off_diagonal_term += coupling * C[n]

            # Full derivative dC_m/dphi
            dC_dphi_vec[m] = dynamical_term - off_diagonal_term
            
        return dC_dphi_vec

    def run_evolution(self, F_path_phi, omega_path_phi, d_phi):
        """
        Calculates the evolution of state coefficients C_n(phi) using RK4
        and computes the non-adiabatic coupling (NAC) magnitude.
        """
        num_bands = self.norbs
        C = np.zeros(num_bands, dtype=complex)
        C[self.initial_band] = 1.0

        C_phi = np.zeros((self.num_phi_steps, num_bands), dtype=complex)
        C_phi[0] = C
        
        nac_magnitudes = np.zeros(self.num_phi_steps)
        data_cache = {}

        for phi_idx in range(self.num_phi_steps - 1):
            F_current = F_path_phi[phi_idx]
            omega_current = omega_path_phi[phi_idx]
            cache_key = (f"{F_current:.6f}", f"{omega_current:.6f}")

            if cache_key not in data_cache:
                energies, states = self.load_floquet_data(F_current, omega_current)
                dHF_dF = self.floquet_model.build_derivative_H(self.k_point, F_current, omega_current, 'epsilon')
                dHF_dO = self.floquet_model.build_derivative_H(self.k_point, F_current, omega_current, 'omega')
                data_cache[cache_key] = (energies, states, dHF_dF, dHF_dO)

            energies_current, states_current, dHF_dF, dHF_dO = data_cache[cache_key]

            dF_dphi = (F_path_phi[phi_idx+1] - F_current) / d_phi if phi_idx < self.num_phi_steps else 0
            dO_dphi = (omega_path_phi[phi_idx+1] - omega_current) / d_phi if phi_idx < self.num_phi_steps else 0
            # --- NAC Calculation ---
            if num_bands >= 2:
                m, n = 1, 0 # Coupling between first excited and ground state
                denom = energies_current[n] - energies_current[m]
                if np.abs(denom) > 1e-9:
                    phi_m, phi_n = states_current[m], states_current[n]
                    
                    vdot_F = np.vdot(phi_m, dHF_dF @ phi_n)
                    vdot_O = np.vdot(phi_m, dHF_dO @ phi_n)
                    coupling_numerator = dF_dphi * vdot_F + dO_dphi * vdot_O
                    
                    # total_nac_phi = coupling_numerator / denom
                    nac_magnitudes[phi_idx] = np.abs(coupling_numerator)

            # --- RK4 Evolution ---
            f_C = lambda C_vec: self._calculate_dC_dphi(
                C_vec, energies_current, states_current, omega_current,
                dHF_dF, dHF_dO, dF_dphi, dO_dphi
            )

            k1 = f_C(C)
            k2 = f_C(C + 0.5 * d_phi * k1)
            k3 = f_C(C + 0.5 * d_phi * k2)
            k4 = f_C(C + d_phi * k3)

            C += (d_phi / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            C_phi[phi_idx + 1] = C
            
        return C_phi, nac_magnitudes

    def load_floquet_data(self, F, omega):
        fname = os.path.join(self.floquet_model.data_path, f"band_E{F:.6f}_O{omega:.6f}.h5")
        try:
            energies_k, states_k_list = load_data_from_hdf5(fname)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found for (F, omega)=({F:.6f}, {omega:.6f}). Please run pre-computation.")
        
        return energies_k[0, :], [s[0, :] for s in states_k_list]

def save_evolution_data(filename, C_phi, nac_magnitudes):
    """Saves time evolution data to an HDF5 file."""
    with h5py.File(filename, 'w') as f:
        f.create_dataset('C_phi', data=C_phi)
        f.create_dataset('nac_magnitudes', data=nac_magnitudes)

def load_evolution_data(filename):
    """Loads time evolution data from an HDF5 file."""
    with h5py.File(filename, 'r') as f:
        C_phi = f['C_phi'][:]
        nac_magnitudes = f['nac_magnitudes'][:]
    return C_phi, nac_magnitudes

if __name__ == "__main__":
    # --- Simulation Setup ---
    coords = [[0], [0.75]]
    tb_model = TightBinding(coords, lambda_decay=1.0, lattice_constant=[1.0], relative_Hopping=[1.5, 1])
    total_time, num_time_steps, nt_assigned = 10, 10001, 11
    
    # --- Define Pulse in Time Domain ---
    t_space = np.linspace(-total_time/2, total_time/2, num_time_steps)
    F_max, t0, sigma = 200.0, 0, total_time / 10.0
    F_t = F_max * np.exp(-((t_space - t0)**2) / (2 * sigma**2))
    omega_initial, omega_final = 12.0, 12.0
    omega_t = np.linspace(omega_initial, omega_final, num_time_steps)

    # --- Change of Variables to Phase Domain ---
    phi_t = cumulative_trapezoid(omega_t, t_space, initial=0)
    phi_final = phi_t[-1]
    phi_path = np.linspace(0, phi_final, num_time_steps)
    d_phi = phi_path[1] - phi_path[0]
    
    t_of_phi_func = interp1d(phi_t, t_space, fill_value="extrapolate")
    F_of_t_func = interp1d(t_space, F_t, fill_value="extrapolate")
    omega_of_t_func = interp1d(t_space, omega_t, fill_value="extrapolate")
    
    t_path_phi = t_of_phi_func(phi_path)
    F_path_phi = F_of_t_func(t_path_phi)
    omega_path_phi = omega_of_t_func(t_path_phi)
    
    # --- K-space Scan Setup ---
    num_k_points = 51 # Reduced for faster testing
    k_grid = np.linspace(-np.pi, 0, num_k_points)
    final_excited_populations = []
    
    # --- Main Loop over k-points ---
    for k_val in k_grid:
        print(f"--- Processing k = {k_val:.4f} ---")
        
        # --- Setup Directories ---
        main_dir = 'MacBook_local_data/chirp_dynamics'
        data_path_k = os.path.join(main_dir, f'chirp_data_k_{k_val:.4f}')
        evolution_cache_dir = os.path.join(main_dir, f'evolution_cache_E_max_{F_max}_omega_{omega_initial}_{omega_final}_total_time_{total_time}_nt_{nt_assigned}')
        output_plot_dir = os.path.join(main_dir, f'dynamics_plots_E_max_{F_max}_omega_{omega_initial}_{omega_final}_total_time_{total_time}_nt_{nt_assigned}')
        os.makedirs(data_path_k, exist_ok=True)
        os.makedirs(evolution_cache_dir, exist_ok=True)
        os.makedirs(output_plot_dir, exist_ok=True)

        # --- Check for Cached Dynamics Data ---
        cache_filename = os.path.join(evolution_cache_dir, f"evolution_k_{k_val:.4f}.h5")
        if os.path.exists(cache_filename):
            print(f"Loading cached evolution data from {cache_filename}")
            C_phi, nac_magnitudes = load_evolution_data(cache_filename)
        else:
            # --- Floquet Pre-computation ---
            floquet_model = tb_model.Floquet(
                E0=F_path_phi, omegad=omega_path_phi, nt=nt_assigned,
                polarization=[1], data_path=data_path_k
            )
            floquet_model.run_dynamics(k=[k_val], E_path=F_path_phi, omega_path=omega_path_phi)
            
            # --- Run Dynamics ---
            print("Running dynamics...")
            time_evolver = TimeEvolution(floquet_model, num_phi_steps=num_time_steps)
            C_phi, nac_magnitudes = time_evolver.run_evolution(F_path_phi, omega_path_phi, d_phi)
            
            # --- Save Dynamics Data ---
            save_evolution_data(cache_filename, C_phi, nac_magnitudes)
            print(f"Saved new evolution data to {cache_filename}")

        final_population_excited_state = np.abs(C_phi[-1, 1])**2
        final_excited_populations.append(final_population_excited_state)
        print(f"Final excited population for k={k_val:.4f}: {final_population_excited_state:.6f}")

        # --- Plotting Individual Dynamics ---
        plot_filename = os.path.join(output_plot_dir, f"dyn_chirp_k{k_val:.4f}.png")
        if not os.path.exists(plot_filename):
            fig, (ax1, ax3) = plt.subplots(
                nrows=2, ncols=1, figsize=(12, 10), sharex=True,
                gridspec_kw={'height_ratios': [3, 2]}
            )
            fig.suptitle(f'Chirped Pulse Evolution (k={k_val:.4f})', fontsize=16, y=0.97)

            color_pop = 'tab:blue'
            ax1.set_ylabel('Population $|C_n|^2$', color=color_pop)
            ax1.plot(t_path_phi, np.abs(C_phi[:, 0])**2, label='State 0')
            ax1.plot(t_path_phi, np.abs(C_phi[:, 1])**2, label='State 1')
            ax1.tick_params(axis='y', labelcolor=color_pop)
            ax1.legend(loc='upper left'); ax1.set_ylim(0, 1.05)
            ax2 = ax1.twinx()
            color_field = 'green'
            ax2.set_ylabel('Field Amplitude F(t)', color=color_field)
            ax2.plot(t_path_phi, F_path_phi, color=color_field, linestyle=':', label='F(t)')
            ax2.tick_params(axis='y', labelcolor=color_field)

            color_nac = 'tab:purple'
            ax3.set_ylabel('NAC Magnitude $|A_{10}(t)|$', color=color_nac)
            ax3.plot(t_path_phi, nac_magnitudes, color=color_nac)
            ax3.tick_params(axis='y', labelcolor=color_nac); ax3.set_yscale('log')
            ax3.set_xlabel('Time (t)')
            ax4 = ax3.twinx()
            color_freq = 'tab:red'
            ax4.set_ylabel('Frequency $\omega(t)$', color=color_freq)
            ax4.plot(t_path_phi, omega_path_phi, color=color_freq, linestyle='--')
            ax4.tick_params(axis='y', labelcolor=color_freq)

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(plot_filename)
            plt.close(fig)
            print(f"Saved dynamics plot to {plot_filename}")
        
        gc.collect()

    # --- Final Summary Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(k_grid, final_excited_populations, marker='o', markersize=4, linestyle='-')
    plt.xlabel('Crystal Momentum (k)')
    plt.ylabel('Final Excitation Population')
    plt.title(f'Final Population vs. k (Chirped Pulse)')
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.grid(True)
    plt.ylim(bottom=0)
    final_plot_path = os.path.join(main_dir, f"final_population_vs_k'dynamics_plots_E_max_{F_max}_omega_{omega_initial}_{omega_final}_total_time_{total_time}_nt_{nt_assigned}'.png")
    plt.savefig(final_plot_path)
    print(f"\nSaved final summary plot to {final_plot_path}")
    plt.show()

