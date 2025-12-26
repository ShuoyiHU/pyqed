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
from  pyqed.floquet.floquet import TightBinding, FloquetBloch
from numpy import exp, eye, zeros, arctan2
from scipy.linalg import eigh
import h5py
import math
from scipy.integrate import solve_ivp
from opt_einsum import contract 
import matplotlib.animation as animation 
from scipy.special import gamma
from scipy.integrate import quad
from scipy.special import j0 # Bessel function of the first kind, order 0 (J₀)

def setup_overlap_static_tensor(FCFM):
    """
    Pre-calculates the time-independent part of the overlap matrix S.
    This should be called once before the time evolution begins.

    Parameters
    ----------
    FCFM : ndarray, shape (n_modes, nt, norbs)
        The Fourier components of the Floquet modes.

    Returns
    -------
    time_independent_tensor : ndarray, shape (n_modes, n_modes, nt, nt)
    """
    # The formula for the overlap matrix is S_ij = Σ_{c,d} P_cd * (Σ_o FCFM*_ico * FCFM_jdo)
    # This pre-calculates the inner part, which is time-independent.
    # Labels:
    # FCFM_conj -> 'ico' (i=mode_idx, c=fourier_idx, o=orbital_idx)
    # FCFM      -> 'jco' (j=mode_idx, c=fourier_idx, o=orbital_idx)
    # Output    -> 'ij'
    print("Pre-calculating time-independent part of the overlap matrix S...")
    norbs = FCFM.shape[2]
    nt = FCFM.shape[1]
    nE_basis = FCFM.shape[0] // norbs
    matrix_A = FCFM.reshape(nE_basis*norbs, nt*norbs)
    # print(matrix_A)
    # print(matrix_A.conj() @ matrix_A.T)  # Debugging line
    # sys.exit()  # Debugging line
    return matrix_A.conj() @ matrix_A.T


if __name__ == '__main__':
    """
    Main execution block to set up and run a quantum dynamics simulation.
    The electric field amplitude E0 follows a "round-trip" path, and the basis
    is constructed from Floquet modes at multiple points along this path.
    Includes functionality to save and load simulation results to avoid re-calculation.
    """
    from scipy.integrate import solve_ivp
    import time
    import os
    import h5py

    # =========================================================================
    #   STEP 1: DEFINE THE SYSTEM AND ESTABLISH THE EXPANDED FLOQUET BASIS
    # =========================================================================
    print("--- STEP 1: Setting up the system and calculating the expanded basis ---")
    
    # --- System Parameters ---
    omega = 10.0
    E_start = 0
    E_peak = 37.5
    nt = 11
    # k_point = [np.pi-0.0001]
    k_point = [[np.pi-0.0001,0]]
    # k_point = [[0.0001,0]]
    b = 0.7
    c = 0.05
    # --- Parameter for the expanded basis ---
    nE_basis = 3 # Number of E-field points to build the basis from
    
    # --- Model Instantiation ---
    coords = [[0.0], [b]]
    # coords = [[0.0,0.0], [b,c]]
    tb_model = TightBinding(coords, relative_Hopping=[1.5, 1.0], lattice_constant=[1])
    floquet_system_basis = tb_model.Floquet(
        omegad=omega, 
        E0=E_start, # E0 here is just a placeholder for the class
        nt=nt, 
        polarization=[1.0],
        data_path=f'MacBook_local_data/temp_data/b_{b:.2f}/'
    )
    
    # --- Create the grid of E-field values for the basis ---
    # E_basis_set = np.linspace(0, 18.7, nE_basis)
    E_basis_set = np.linspace(0, 37.5, nE_basis)
    print(f"Constructing basis from {nE_basis} E-field points: {np.round(E_basis_set, 2)}")

    # --- Loop through the E-field grid to build the full basis ---
    all_quasi_energies = np.zeros(nE_basis*floquet_system_basis.norbs, dtype=complex)
    basis_fcfm = np.zeros((nE_basis*floquet_system_basis.norbs, nt, floquet_system_basis.norbs), dtype=complex)

    previous_evecs_NF_by_norbs = None 

    for i, E_val in enumerate(E_basis_set):
        print(f"  Calculating modes for E0 = {E_val:.2f}...")
        energies, evecs_list_raw = floquet_system_basis.track_Floquet_Modes(
            k_values=k_point, 
            E0=E_val
        )
        print(f"  Found {energies} quasi-energies for E0 = {E_val:.2f}")
        if evecs_list_raw[0].size == 0:
            print(f"Error: Could not calculate basis at E0 = {E_val}. Exiting.")
            sys.exit()
        for mode_idx in range(floquet_system_basis.norbs):
            global_idx = i * floquet_system_basis.norbs + mode_idx
            basis_fcfm[global_idx] = evecs_list_raw[mode_idx]
            all_quasi_energies[global_idx] = energies[mode_idx]
    print(np.shape(basis_fcfm))

    time_independent_S_tensor = setup_overlap_static_tensor(basis_fcfm)
    E_diag_matrix = np.tile(all_quasi_energies, (len(all_quasi_energies), 1))
    # E_diag_matrix = np.diag(all_quasi_energies)
    print(E_diag_matrix)
    # sys.exit()
    # =========================================================================
    #   STEP 2: DEFINE THE DYNAMICS (PATH AND INITIAL STATE)
    # =========================================================================
    print("\n--- STEP 2: Defining the dynamics path and initial conditions ---")

    # --- Gaussian pulse parameters ---
    total_time = 50
    t0 = 0
    sigma = total_time / 10.0
    t_space = np.linspace(-total_time / 2, total_time / 2, 5001)

    def E_path(t):
        """
        Defines the Gaussian pulse electric field.
        """
        # Ensure that E_peak is defined in the main block
        return E_peak * np.exp(-((t - t0)**2) / (2 * sigma**2))
    num_modes = nE_basis * floquet_system_basis.norbs
    C0 = np.zeros(num_modes, dtype=complex)
    # Set all odd-indexed components of C0 to the same value, even-indexed to zero, then normalize
    odd_indices = np.arange(1, num_modes, 2)
    C0[:] = 0
    if len(odd_indices) > 0:
        C0[odd_indices-1] = 1.0
        C0 /= np.linalg.norm(C0)
    print(C0)
    print(f"System will follow a Gaussian pulse: E_max = {E_peak}, total_time = {total_time}, sigma = {sigma}")
    
    # =========================================================================
    #   STEP 3: BUILD AND RUN THE EOM SOLVER 
    # =========================================================================
    print("\n--- STEP 3: Building and running the EOM solver ---")

    # --- Define a unique filename for the results cache ---
    results_dir = "MacBook_local_data/simulation_results_total_static_basis"
    os.makedirs(results_dir, exist_ok=True)
    results_filename = os.path.join(results_dir, f"dynamics_nE{nE_basis}_Epeak{E_peak:.1f}_Tramp{total_time:.1f}_k{k_point[0]}.h5")
    
    solution = None
    time_taken_per_logged_interval = None 
    logged_simulation_times_for_speed_plot = None 
    logged_simulation_times_for_norms = None 
    logged_norm_SE_diag = None 
    logged_norm_M_t = None 
    logged_M_t_matrices = None 
    logged_S_cond_num = None 

    if os.path.exists(results_filename):
        print(f"Loading previously calculated dynamics data from: {results_filename}")
        with h5py.File(results_filename, 'r') as f:
            from types import SimpleNamespace
            solution = SimpleNamespace()
            solution.t = f['t'][:]
            solution.y = f['y'][:]
            solution.status = 0 
            
            if 'logged_real_times' in f and 'logged_sim_times' in f:
                logged_real_times_loaded = f['logged_real_times'][:]
                logged_simulation_times_full = f['logged_sim_times'][:] 
                if len(logged_real_times_loaded) > 1:
                    time_taken_per_logged_interval = np.diff(logged_real_times_loaded)
                    logged_simulation_times_for_speed_plot = logged_simulation_times_full[:-1] 
                    logged_simulation_times_for_norms = logged_simulation_times_full 
                else:
                    time_taken_per_logged_interval = None
                    logged_simulation_times_for_speed_plot = None
                    logged_simulation_times_for_norms = None
                print("Loaded logged real times and simulation times for speed plot.")
            else:
                print("Warning: Logged speed data (real/sim times) not found in loaded HDF5 file.")
            
            if 'logged_norm_SE_diag' in f and 'logged_norm_M_t' in f:
                logged_norm_SE_diag = f['logged_norm_SE_diag'][:]
                logged_norm_M_t = f['logged_norm_M_t'][:]
                if logged_simulation_times_for_norms is None and 'logged_sim_times' in f:
                    logged_simulation_times_for_norms = f['logged_sim_times'][:]
                print("Loaded logged norms of SE_diag and M_t.")
            else:
                print("Warning: Logged norm data (SE_diag and M_t) not found in loaded HDF5 file.")
            
            if 'logged_M_t_matrices' in f:
                logged_M_t_matrices = f['logged_M_t_matrices'][:]
                if logged_simulation_times_for_norms is None and 'logged_sim_times' in f:
                    logged_simulation_times_for_norms = f['logged_sim_times'][:]
                print("Loaded logged M_t matrices for heatmap video.")
            else:
                print("Warning: Logged M_t matrices data not found in loaded HDF5 file.")

            if 'logged_S_cond_num' in f: 
                logged_S_cond_num = f['logged_S_cond_num'][:]
                if logged_simulation_times_for_norms is None and 'logged_sim_times' in f:
                    logged_simulation_times_for_norms = f['logged_sim_times'][:]
                print("Loaded logged S_t condition numbers.")
            else:
                print("Warning: Logged S_t condition number data not found in loaded HDF5 file.")

    else:
        print(f"No cached data found. Starting numerical integration...")
        E_basis_set_for_coupling = E_basis_set

        global _step_counter
        _step_counter = 0
        global _log_real_times
        _log_real_times = []
        global _log_sim_times
        _log_sim_times = []
        global _log_norm_SE_diag
        _log_norm_SE_diag = []
        global _log_norm_M_t
        _log_norm_M_t = []
        global _log_M_t_matrices
        _log_M_t_matrices = []
        global _log_S_cond_num 
        _log_S_cond_num = []
        
        log_interval = 10 

        def dynamics_eom_rhs(t, C, model, E_diag, basis_fcfm_arg, k_pt, static_S_tensor):
            global _step_counter
            global _log_real_times
            global _log_sim_times
            global _log_norm_SE_diag
            global _log_norm_M_t
            global _log_M_t_matrices
            global _log_S_cond_num 
            global start_time 

            _step_counter += 1
            
            S_t = static_S_tensor
            # S_t = model.overlap_matrix_of_Floquet_modes(static_S_tensor, t)
            M_t = model.Coupling_matrix(basis_fcfm_arg, k_pt, E_path(t), E_basis_set_for_coupling, t)

            try:
                cond_num = np.linalg.cond(S_t)
                S_inv_t = np.linalg.pinv(S_t) 
                
            except linalg.LinAlgError:
                print(f"ERROR: linalg.pinv encountered an issue with S(t) at time t={t}. Halting.")
                cond_num = 1e100 # Indicate severe problem
                return np.zeros_like(C)
            print(np.shape(S_t), np.shape(M_t), np.shape(E_diag), np.shape(C))
            H_eff_term = (S_t @ E_diag) + M_t
            asym = np.max(np.abs(H_eff_term + H_eff_term.conj().T))
            print("Hermiticity of H_eff:", asym)    
            # sys.exit()
            hbar = 1.0
            dC_dt = (1 / (1j * hbar)) * (S_inv_t @ H_eff_term @ C)
            
            if _step_counter % log_interval == 1: 
                _log_real_times.append(time.time())
                _log_sim_times.append(t)
                _log_norm_SE_diag.append(np.linalg.norm(S_t @ E_diag))
                _log_norm_M_t.append(np.linalg.norm(M_t))
                _log_M_t_matrices.append(np.abs(M_t))
                _log_S_cond_num.append(cond_num) 

            print(f"t={t:.4f}, E(t)={E_path(t):.4f}, time_taken={time.time() - start_time:.4f} seconds") 
            return dC_dt

        t_final = total_time / 2 # Corrected t_final and t_initial for solve_ivp
        t_initial = -total_time / 2
        t_span = (t_initial, t_final)
        
        solver_args = (floquet_system_basis, E_diag_matrix, basis_fcfm, k_point, time_independent_S_tensor)
        
        start_time = time.time() 
        solution = solve_ivp(
            fun=dynamics_eom_rhs,
            t_span=t_span,
            y0=C0,
            args=solver_args,
            method='RK45',
            rtol=1e-6, # You can uncomment and tighten these if norm conservation is still an issue
            atol=1e-8  # after using pinv and phase alignment.
        )
        end_time = time.time()
        print(f"Integration complete. Total time elapsed: {end_time - start_time:.2f} seconds.")

        if len(_log_real_times) > 1:
            time_taken_per_logged_interval = np.diff(_log_real_times)
            logged_simulation_times_for_speed_plot = np.array(_log_sim_times[:-1]) 
            logged_simulation_times_for_norms = np.array(_log_sim_times) 
            
            logged_norm_SE_diag = np.array(_log_norm_SE_diag)
            logged_norm_M_t = np.array(_log_norm_M_t)
            logged_M_t_matrices = np.array(_log_M_t_matrices) 
            logged_S_cond_num = np.array(_log_S_cond_num) 

        else:
            time_taken_per_logged_interval = None
            logged_simulation_times_for_speed_plot = None
            logged_simulation_times_for_norms = None
            logged_norm_SE_diag = None
            logged_norm_M_t = None
            logged_M_t_matrices = None
            logged_S_cond_num = None 
            print("Warning: Not enough logged data to calculate time per interval or norms/matrices.")
            
        if solution.status == 0:
            print(f"Saving dynamics data to: {results_filename}")
            with h5py.File(results_filename, 'w') as f:
                f.create_dataset('t', data=solution.t)
                f.create_dataset('y', data=solution.y)
                if time_taken_per_logged_interval is not None:
                    f.create_dataset('logged_real_times', data=np.array(_log_real_times))
                    f.create_dataset('logged_sim_times', data=np.array(_log_sim_times)) 
                    f.create_dataset('logged_norm_SE_diag', data=logged_norm_SE_diag)
                    f.create_dataset('logged_norm_M_t', data=logged_norm_M_t)
                    f.create_dataset('logged_M_t_matrices', data=logged_M_t_matrices) 
                    f.create_dataset('logged_S_cond_num', data=logged_S_cond_num) 
        else:
            print(f"Solver failed with status {solution.status}: {solution.message}")


    # =========================================================================
    #   STEP 4: DIAGNOSE AND VISUALIZE THE RESULTS
    # =========================================================================
    if solution is not None and solution.y.size > 1:
        print("\n--- STEP 4: Diagnosing and Visualizing the results ---")

        # --- Diagnostic Checks ---
        initial_pops = np.abs(solution.y[:, 0])**2
        final_pops = np.abs(solution.y[:, -1])**2
        print(f"Initial total population: {np.sum(initial_pops):.6f}")
        print(f"Final total population:   {np.sum(final_pops):.6f}")
        print(f"Final total population (excluding state 0): {np.sum(final_pops[1:]):.6f}")
        if not np.allclose(initial_pops[0], 1.0):
            print(f"WARNING: Initial population of state 0 is {initial_pops[0]:.6f}, but expected 1.0.")
            print("Initial population vector:", np.round(initial_pops, 3))
        
        if solution.status != 0:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"!!! WARNING: Solver failed. {solution.message}")
            if solution.t.size > 1:
                print(f"!!! Plotting only the partial results up to t = {solution.t[-1]:.4f}")
            else:
                print("!!! Plotting only the initial state at t = 0.")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

        # --- Plotting Main Figure ---
        populations = np.abs(solution.y)**2
        
        fig1, ax1 = plt.subplots(figsize=(12, 7)) 
        
        color = 'tab:blue'
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Population', fontsize=12, color=color)
        
        ax1.plot(solution.t, populations[0, :], label=f'Pop. of State 0 (from E₀={E_basis_set[0]:.1f})')
        ax1.plot(solution.t, populations[1, :], label=f'Pop. of State 1 (from E₀={E_basis_set[0]:.1f})')
        
        if num_modes > 2:
            other_pops = np.sum(populations[2:, :], axis=0)
            ax1.plot(solution.t, other_pops, label='Pop. of all other basis states', linestyle=':')

        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        color_E = 'tab:red'
        ax2.set_ylabel('E-Field Amplitude E₀(t)', fontsize=12, color=color_E)
        ax2.plot(solution.t, [E_path(t) for t in solution.t], color=color_E, linestyle='--', label='E₀(t) Path')
        ax2.tick_params(axis='y', labelcolor=color_E)
        
        all_lines_fig1 = [] 
        all_labels_fig1 = [] 

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        all_lines_fig1.extend(lines1)
        all_labels_fig1.extend(labels1)
        all_lines_fig1.extend(lines2)
        all_labels_fig1.extend(labels2)

        if logged_norm_SE_diag is not None and logged_norm_M_t is not None and logged_simulation_times_for_norms is not None:
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('outward', 60)) 
            color_norms_SE = 'tab:purple'
            color_norms_M = 'tab:orange'
            ax3.set_ylabel('Norm of Operators', fontsize=12, color=color_norms_SE)

            ax3.plot(logged_simulation_times_for_norms, logged_norm_SE_diag, color=color_norms_SE, linestyle='-', marker='x', markersize=4, label=f'Norm(S@E_diag)')
            ax3.plot(logged_simulation_times_for_norms, logged_norm_M_t, color=color_norms_M, linestyle='-', marker='o', markersize=4, label=f'Norm(M_t)')
            
            ax3.tick_params(axis='y', labelcolor=color_norms_SE)
            lines3, labels3 = ax3.get_legend_handles_labels()
            all_lines_fig1.extend(lines3)
            all_labels_fig1.extend(labels3)

        if time_taken_per_logged_interval is not None and logged_simulation_times_for_speed_plot is not None:
            ax4 = ax1.twinx()
            ax4.spines['right'].set_position(('outward', 120)) 
            color_speed = 'tab:green'
            ax4.set_ylabel(f'Time per {log_interval} steps (s)', fontsize=12, color=color_speed) 

            ax4.plot(logged_simulation_times_for_speed_plot, time_taken_per_logged_interval, 
                     color=color_speed, linestyle=':', marker='.', markersize=4, label=f'Time per {log_interval} steps')

            ax4.tick_params(axis='y', labelcolor=color_speed)
            lines4, labels4 = ax4.get_legend_handles_labels()
            all_lines_fig1.extend(lines4)
            all_labels_fig1.extend(labels4)

        ax1.legend(all_lines_fig1, all_labels_fig1, loc='upper center', bbox_to_anchor=(0.5, 1.25), 
                   ncol=4, fancybox=True, shadow=True, fontsize=9) 


        fig1.suptitle('Floquet State Population Dynamics During E-Field Pulse', fontsize=16) 
        plt.tight_layout(rect=[0, 0, 1, 0.85]) 
        plt.show()

        # =========================================================================
        # NEW EXTRA PLOT: Population of Each Individual State
        # =========================================================================
        print("\n--- Generating plot for individual state populations ---")
        fig_individual_pops, ax_individual_pops = plt.subplots(figsize=(10, 6))
        norbs = floquet_system_basis.norbs
        # Loop through all modes and plot their populations
        for i in range(num_modes):
            # Assign a color and label based on its E_basis_set origin
            e_origin_idx = i // norbs # Which E_basis_set index this mode comes from
            e_origin_val = E_basis_set[e_origin_idx]
            
            # Label might be "State X (from E=Y.Z)"
            ax_individual_pops.plot(solution.t, populations[i, :], 
                                    label=f'Pop. State {i} (E={e_origin_val:.1f})')

        ax_individual_pops.set_xlabel('Time', fontsize=12)
        ax_individual_pops.set_ylabel('Population', fontsize=12)
        ax_individual_pops.set_yscale('log')
        ax_individual_pops.set_title('Population of Each Basis State Over Time', fontsize=14)
        ax_individual_pops.grid(True)
        # Place legend outside if too many lines, or hide for dense plots
        if num_modes <= 30: # Only show legend if not too cluttered
            ax_individual_pops.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend on side
        else:
            print("  Too many states to display legend clearly. Legend omitted for individual population plot.")
            plt.tight_layout()
            
        plt.show()


        # =========================================================================
        # NEW: Plot S_t Condition Number
        # =========================================================================
        if logged_S_cond_num is not None and logged_simulation_times_for_norms is not None:
            print("\n--- Plotting S_t Condition Number ---")
            fig_cond, ax_cond = plt.subplots(figsize=(10, 6))

            ax_cond.semilogy(logged_simulation_times_for_norms, logged_S_cond_num, 
                             color='blue', linestyle='-', marker='.', markersize=4,
                             label='Condition Number of S(t)')
            
            ax_cond.set_xlabel('Time', fontsize=12)
            ax_cond.set_ylabel('Condition Number of S(t)', fontsize=12, color='blue')
            ax_cond.tick_params(axis='y', labelcolor='blue')
            ax_cond.set_title('Condition Number of Overlap Matrix S(t) Over Time', fontsize=14)
            ax_cond.grid(True, which="both", ls="-")
            ax_cond.legend(loc='upper left')
            plt.tight_layout()
            plt.show()


        # =========================================================================
        # NEW: Generate Heatmap Video of |M_t|
        # =========================================================================
        if logged_M_t_matrices is not None and logged_simulation_times_for_norms is not None:
            print("\n--- Generating Heatmap Frames ---")
            
            # Directories
            frames_dir_M = os.path.join(results_dir, f"M_t_heatmap_frames_nE{nE_basis}_Epeak{E_peak:.1f}_T{total_time:.1f}")
            os.makedirs(frames_dir_M, exist_ok=True)

            # Color scaling
            all_M_elements = logged_M_t_matrices.flatten()
            vmin_M, vmax_M = np.percentile(all_M_elements, 5), np.percentile(all_M_elements, 95)
            if vmax_M == vmin_M: vmax_M += 1e-9

            # Generate frames
            for frame_idx, t_val in enumerate(logged_simulation_times_for_norms):
                current_E_field = E_path(t_val)

                # Coupling M_t heatmap
                figM, axM = plt.subplots(figsize=(6, 5))
                imM = axM.imshow(logged_M_t_matrices[frame_idx], cmap='hot', origin='lower',
                                vmin=vmin_M, vmax=vmax_M)
                figM.colorbar(imM, ax=axM, label='|M_ij(t)|')
                axM.set_title(f'|M_ij| at t={t_val:.3f}, E={current_E_field:.2f}')
                plt.tight_layout()
                figM.savefig(os.path.join(frames_dir_M, f"frame_{frame_idx:04d}.png"), dpi=120)
                plt.close(figM)


            print("\nHeatmap frames saved.")
            print("To generate videos, run in terminal:")
            print(f"cd {frames_dir_M} && ffmpeg -framerate 10 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p ../M_t_animation.mp4")