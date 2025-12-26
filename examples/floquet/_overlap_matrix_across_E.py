import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import gc
from pyqed.floquet.floquet import TightBinding
from pyqed.floquet.utils import load_data_from_hdf5


def test_Gomez_Leon_2013(E0=30, number_of_step_in_b=51, nt=11, slice_b_value=None):
    """
    Runs the main simulation to calculate Floquet bands over a parameter grid.
    Saves all energy and state data to .h5 files.
    """
    print("--- Starting Main Simulation: test_Gomez_Leon_2013 ---")
    
    # 1. Parameters Setup
    omega = 10
    E_over_omega = np.linspace(0, E0 / omega, 501)
    E = [e * omega for e in E_over_omega]
    num_k = 201
    k_vals = np.stack([np.linspace(-np.pi, np.pi, num_k), np.zeros(num_k)], axis=1)
    b_grid = [0.70]  # Using a single value as in your example
    c_grid = [0.0]
    # c_grid = np.linspace(0, 0.1, 11)
    data_path = "" # Initialize path variable

    # 2. Main Calculation Loop
    for c in c_grid:
        for b_idx, b in enumerate(b_grid):
            print(f"--- Calculating for b = {b:.4f}, c = {c:.4f} ---")
            
            # This is the path where data for this run will be saved
            data_path = f'MacBook_local_data/for_plot/chiral_k={num_k:.1f}_nt_5_new/c={c:.4f}/floquet_data_Gomez_Leon_test_b={b:.2f}/'
            
            coords = [[0, 0], [b, c]]
            tb_model = TightBinding(coords, lambda_decay=1.0, lattice_constant=[1.0, 0], nk=100, mu=0.0, relative_Hopping=[1.5, 1])
            
            floquet_model = tb_model.Floquet(omegad=omega, E0=E, nt=nt, polarization=[1, 1j], data_path=data_path)
            
            # This is the main computational step that saves all .h5 files
            energies, states = floquet_model.run(k_vals)
            
            print(f"Finished calculation for b={b:.2f}. Data saved in {data_path}")
            gc.collect()

    print("--- Main Simulation Finished ---")

    # --- MODIFICATION ---
    # Return the key parameters from the final loop iteration.
    return E, k_vals, data_path

# --- POST-PROCESSING AND ANALYSIS FUNCTION ---
# This function loads the saved data to perform the specific analysis.
def analyze_and_plot_overlap_across_E(k_fixed, E_values, k_values, data_path):
    """
    Post-processes saved data to calculate and plot the overlap matrix
    for a fixed k-point across a range of E values.
    """
    print("\n--- Starting Post-Processing Analysis ---")
    print(f"Analyzing overlap matrix at fixed k = {k_fixed:.4f}")

    # 1. Find the index corresponding to the fixed k-point
    try:
        k_idx = np.argmin(np.abs(k_values[:, 0] - k_fixed))
        actual_k = k_values[k_idx, 0]
        print(f"Requested k={k_fixed:.4f}, found closest k-point at index {k_idx} (k={actual_k:.4f})")
    except IndexError:
        print("Error: k_values array seems to have the wrong shape. Exiting.")
        return

    # 2. Loop through all E values to load the state vectors at the fixed k_idx
    ground_states_at_k = []
    excited_states_at_k = []

    for E_val in E_values:
        fname = os.path.join(data_path, f"band_E{E_val:.6f}.h5")
        try:
            _, all_bands_states = load_data_from_hdf5(fname)
            ground_states_at_k.append(all_bands_states[0][k_idx, :])
            excited_states_at_k.append(all_bands_states[1][k_idx, :])
        except FileNotFoundError:
            print(f"Warning: Data file not found for E={E_val:.6f}. Skipping this point.")
            continue
            
    if not ground_states_at_k:
        print("Error: No data was loaded. Check data_path and E_values.")
        return

    # 3. Calculate the overlap matrix
    ground_states = np.array(ground_states_at_k)
    excited_states = np.array(excited_states_at_k)
    overlap_matrix = np.matmul(ground_states, excited_states.conj().T)
    print(f"Computed overlap matrix of shape {overlap_matrix.shape}")

    # 4. Generate the heatmap
    plt.figure(figsize=(8, 7))
    im = plt.imshow(np.abs(overlap_matrix), origin='upper', aspect='equal', cmap='viridis', interpolation='nearest')
    
    plt.title(f"Overlap Matrix Heatmap at k = {actual_k:.4f}", fontsize=14)
    plt.xlabel("E-point index (j)", fontsize=12)
    plt.ylabel("E-point index (i)", fontsize=12)
    
    cbar = plt.colorbar(im)
    cbar.set_label(r'$|<\psi_g(E_i)|\psi_e(E_j)>|$', fontsize=12)
    
    plot_dir = os.path.join(data_path, "overlap_plots_fixed_k")
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, f"overlap_heatmap_k={actual_k:.4f}.png")
    
    plt.savefig(save_path, dpi=300)
    print(f"Saved overlap heatmap to {save_path}")
    plt.show()



# --- MAIN EXECUTION BLOCK ---
# run simulation, then run analysis.
if __name__ == "__main__":
    
    # --- Step 1: Run your simulation function ---
    E_list, k_grid, simulation_data_path = test_Gomez_Leon_2013()

    # --- Step 2: Run the post-processing analysis ---
    # k_to_analyze = np.pi/2  # Example: The Gamma point (k=0) where gaps often close
    k_to_analyze = np.pi  # Example: The Gamma point (k=0) where gaps often close
    # k_to_analyze = 0  # Example: The Gamma point (k=0) where gaps often close

    analyze_and_plot_overlap_across_E(
        k_fixed=k_to_analyze,
        E_values=E_list,
        k_values=k_grid,
        data_path=simulation_data_path
    )