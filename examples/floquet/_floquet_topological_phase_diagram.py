import numpy as np
import matplotlib.pyplot as plt
from  pyqed.floquet.floquet import TightBinding, FloquetBloch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from pyqed.floquet.utils import track_valence_band, berry_phase_winding, figure, track_valence_band_GL2013, save_data_to_hdf5, load_data_from_hdf5
import matplotlib.colors as colors
import gc 


def test_Gomez_Leon_2013(E0=50, number_of_step_in_b=51, nt=11, slice_b_value=None):
    """
    Calculates winding number and band gap, then plots a correlated slice
    plot and the final winding number heatmap.
    """
    # 1. Parameters Setup
    omega = 10
    E_over_omega = np.linspace(0, E0/omega, 201) # Reduced steps for faster testing
    E = [e * omega for e in E_over_omega]
    num_k = 200  # Reduced k values for faster testing
    # k_vals = np.linspace(-np.pi, np.pi, num_k)
    k_vals = np.stack([np.linspace(-np.pi, np.pi, num_k), np.zeros(num_k)], axis=1)
    b_grid = np.linspace(0, 1, number_of_step_in_b)
    # b_grid = np.linspace(1, 0, number_of_step_in_b)
    b_grid = [0.70]

    # c_grid = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.105, 0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18, 0.185, 0.19, 0.195, 0.200, 0.205, 0.21, 0.215, 0.22, 0.225, 0.23, 0.235, 0.24, 0.245, 0.25, 0.255, 0.26, 0.265, 0.27, 0.275, 0.28, 0.285, 0.29, 0.295, 0.3] # Reduced c_grid for faster testing

    c_grid = [0]
    
    # --- Create grids to store both winding number and band gap data ---
    winding_number_grid = np.zeros((len(b_grid), len(E)), dtype=complex)
    
    min_band_gap_grid = np.zeros((len(b_grid), len(E)))
    max_band_gap_grid = np.zeros((len(b_grid), len(E))) # New grid for max gap
    winding_number_band = 0

    # 2. Main Calculation Loop
    for c in c_grid:
        for b_idx, b in enumerate(b_grid):
            print(f"--- Starting Calculation for b = {b:.4f} ---")
            # Create tight-binding model
            coords = [[0, 0], [b, c]]
            # coords = [[0, 0], [b, c]]
            tb_model = TightBinding(coords, lambda_decay=1.0, lattice_constant=[1.0, 0], nk=100, mu=0.0, relative_Hopping=[1.5,1])
            
            # Initialize Floquet model
            data_path = f'MacBook_local_data/for_plot/chiral_k={num_k:.1f}_nt_5_new/c={c:.4f}/floquet_data_Gomez_Leon_test_b={b:.2f}/'
            floquet_model = tb_model.Floquet(omegad=omega, E0=E, nt=nt, polarization=[1,1j], data_path=data_path)
            
            # Run simulation, which saves all .h5 files to disk
            energies, states = floquet_model.run(k_vals)
            
            # --- Calculate Winding Number (Vectorized) ---
            # winding_number_grid[b_idx] = floquet_model.winding_number(band_id=winding_number_band)
            # winding_number_grid[b_idx] = [min(2-a, a) for a in winding_number_grid[b_idx]]

            print(f"Winding number calculation complete for b={b:.2f}")
            floquet_model.plot_band_structure(k_vals,save_band_structure=True)

            # --- Calculate Minimum and Maximum Band Gaps ---
            print("Calculating min/max band gaps by reading saved files...")
            for e_idx, E_val in enumerate(E):
                fname = os.path.join(data_path, f"band_E{E_val:.6f}.h5")
                try:
                    band_energy, _ = load_data_from_hdf5(fname)
                    if band_energy.shape[1] >= 2:
                        gaps_at_each_k = np.abs(band_energy[:, 1] - band_energy[:, 0])
                        # Store both the minimum and maximum gap
                        min_band_gap_grid[b_idx, e_idx] = np.min(gaps_at_each_k)
                        max_band_gap_grid[b_idx, e_idx] = np.max(gaps_at_each_k)
                except FileNotFoundError:
                    print(f"Warning: Data file not found for E={E_val:.6f}.")
                    min_band_gap_grid[b_idx, e_idx] = np.nan
                    max_band_gap_grid[b_idx, e_idx] = np.nan
            print("Band gap calculation complete.")
            print('')
        print("\n--- Generating Winding Number Heatmap ---")
        B, E_mesh = np.meshgrid(b_grid, E_over_omega)
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(B, E_mesh, winding_number_grid.T.real, shading='auto', cmap='viridis')
        plt.colorbar(label='Winding Number')
        plt.xlabel('Bond Length b')
        plt.ylabel(r'$E_0 / \omega$')
        plt.title(f'Floquet Winding Number Map (Band {winding_number_band})')
        plt.tight_layout()
        save_dir = f'MacBook_local_data/for_plot/chiral_k={num_k:.1f}_nt_5_new/c={c:.4f}_floquet_winding_map.png'
        abs_save_dir = f'MacBook_local_data/for_plot/chiral_k={num_k:.1f}_nt_5_new/absolute_band_gap_map_c={c:.4f}_.png'
        rel_save_dir = f'MacBook_local_data/for_plot/chiral_k={num_k:.1f}_nt_5_new/arelative_band_gap_map_c={c:.4f}.png'
        # save_dir = f'MacBook_local_data/chiral_k={num_k:.1f}_nt_5_new/c={c:.4f}_floquet_winding_map.png'
        # abs_save_dir = f'MacBook_local_data/chiral_k={num_k:.1f}_nt_5_new/absolute_band_gap_map_c={c:.4f}_.png'
        # rel_save_dir = f'MacBook_local_data/chiral_k={num_k:.1f}_nt_5_new/arelative_band_gap_map_c={c:.4f}.png'
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        plt.savefig(save_dir, dpi=300)
        # plt.show()
        # 2. Absolute Band Gap Heatmap
        print("Generating Absolute Band Gap Heatmap...")
        plt.figure(figsize=(8, 6))
        # Use a logarithmic color scale to better visualize small gap values
        pcm = plt.pcolormesh(B, E_mesh, min_band_gap_grid.T, shading='auto', 
                            norm=colors.LogNorm(vmin=min_band_gap_grid[min_band_gap_grid>0].min(), vmax=min_band_gap_grid.max()),
                            cmap='magma')
        plt.colorbar(pcm, label='Minimum Gap (log scale)')
        plt.xlabel('Bond Length b')
        plt.ylabel(r'$E_0 / \omega$')
        plt.title(f'Absolute Band Gap Map (c={c:.2f})')
        plt.tight_layout()
        plt.savefig(abs_save_dir, dpi=300)
        plt.close()

        # 3. Relative Band Gap Heatmap
        print("Generating Relative Band Gap Heatmap...")
        # Calculate the relative gap, handling potential division by zero
        relative_band_gap_grid = np.divide(min_band_gap_grid, max_band_gap_grid, 
                                        out=np.zeros_like(min_band_gap_grid), 
                                        where=max_band_gap_grid!=0)
        plt.figure(figsize=(8, 6))
        pcm = plt.pcolormesh(B, E_mesh, relative_band_gap_grid.T, shading='auto', vmin=0, vmax=1, cmap='inferno')
        plt.colorbar(pcm, label='Min Gap / Max Gap')
        plt.xlabel('Bond Length b')
        plt.ylabel(r'$E_0 / \omega$')
        plt.title(f'Relative Band Gap Map (c={c:.2f})')
        plt.tight_layout()
        plt.savefig(rel_save_dir, dpi=300)
        plt.close()

        # 3. Correlated Slice Plot Generation
        if slice_b_value is not None:
            slice_idx = np.argmin(np.abs(np.array(b_grid) - slice_b_value))
            actual_b_val = b_grid[slice_idx]
            
            # Extract the 1D data for both quantities
            winding_slice_data = winding_number_grid[slice_idx]
            gap_slice_data = min_band_gap_grid[slice_idx]

            print(f"\n--- Generating Correlated Slice Plot for b = {actual_b_val:.4f} ---")
            
            # --- Create a figure with two subplots, sharing the x-axis ---
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            fig.suptitle(f'Topological Properties at Bond Length b = {actual_b_val:.4f}', fontsize=16)

            # --- Subplot 1: Winding Number ---
            ax1.plot(E_over_omega, winding_slice_data.real, 'b.-', markersize=4, label='Winding Number')
            ax1.set_ylabel('Winding Number', fontsize=12, color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_ylim(-0.1, 1.1)
            ax1.set_yticks([0, 0.5, 1.0])
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # --- Subplot 2: Band Gap ---
            ax2.plot(E_over_omega, gap_slice_data, 'r.-', markersize=4, label='Min. Band Gap')
            ax2.set_ylabel('Minimum Band Gap', fontsize=12, color='red')
            ax2.set_xlabel(r'$E_0 / \omega$', fontsize=14)
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.set_ylim(bottom=0)
            # Set y-axis to log scale and autoscale to fit all y values
            ax2.set_yscale('log')
            ax2.autoscale(enable=True, axis='y')

            # Final layout adjustments and saving
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            slice_save_dir = 'MacBook_local_data/for_plot/chiral/'
            os.makedirs(slice_save_dir, exist_ok=True)
            plt.savefig(os.path.join(slice_save_dir, f'correlated_slice_c={c}_b={actual_b_val:.2f}.png'), dpi=300)
            # plt.show()
            plt.clf()
            # --- Create a single figure and axis ---
            fig, ax1 = plt.subplots(figsize=(10, 6))
            fig.suptitle(f'Topological Properties at Bond Length b = {actual_b_val:.4f}', fontsize=16)

            # --- Plot Winding Number on the first Y-axis (left) ---
            color1 = 'tab:blue'
            ax1.set_xlabel(r'$E_0 / \omega$', fontsize=14)
            ax1.set_ylabel('Winding Number', fontsize=12, color=color1)
            p1 = ax1.plot(E_over_omega, winding_slice_data.real, color=color1, linestyle='-', marker='.', markersize=4, label='Winding Number')
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.set_ylim(-0.1, 1.1)
            ax1.set_yticks([0, 0.5, 1.0])
            ax1.grid(True, linestyle='--', alpha=0.7)

            # --- Create a second Y-axis that shares the same X-axis ---
            ax2 = ax1.twinx()
            
            # --- Plot Band Gap on the second Y-axis (right) ---
            color2 = 'tab:red'
            ax2.set_ylabel('Minimum Band Gap', fontsize=12, color=color2)
            p2 = ax2.plot(E_over_omega, gap_slice_data, color=color2, linestyle='--', marker='x', markersize=4, label='Min. Band Gap')
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.set_ylim(bottom=0)

            # --- Create a combined legend ---
            lns = p1 + p2
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='upper left')

            # Final layout adjustments and saving
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            slice_save_dir = 'MacBook_local_data/for_plot/chiral_k=1000.0_nt_5_new/'
            os.makedirs(slice_save_dir, exist_ok=True)
            plt.savefig(os.path.join(slice_save_dir, f'dual_axis_slice_c={c}_b={actual_b_val:.2f}.png'), dpi=300)
            # plt.show()
            plt.clf()

        # 4. Winding Number Heatmap Plot
        print("\n--- Generating Winding Number Heatmap ---")
        B, E_mesh = np.meshgrid(b_grid, E_over_omega)
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(B, E_mesh, winding_number_grid.T.real, shading='auto', cmap='viridis')
        plt.colorbar(label='Winding Number')
        plt.xlabel('Bond Length b')
        plt.ylabel(r'$E_0 / \omega$')
        plt.title(f'Floquet Winding Number Map (Band {winding_number_band})')
        plt.tight_layout()
        save_dir = f'MacBook_local_data/no_reassign_test/chiral_k=200.0_new/c={c_grid[0]:.4f}_floquet_winding_map_0_2.png'
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        plt.savefig(save_dir, dpi=300)
        # plt.show()

def test_Gomez_Leon_2013_realistic_parameter(E0=2, number_of_step_in_b=21, nt=81, slice_b_value=None):
    """
    Calculates winding number and band gap, then plots a correlated slice
    plot and the final winding number heatmap.
    """
    # 1. Parameters Setup
    omega = 0.1
    E_over_omega = np.linspace(0, E0/omega, 51) # Reduced steps for faster testing
    E = [e * omega for e in E_over_omega]
    num_k = 100  # Reduced k values for faster testing
    # k_vals = np.linspace(-np.pi, np.pi, num_k)
    k_vals = np.stack([np.linspace(-np.pi, np.pi, num_k), np.zeros(num_k)], axis=1)
    b_grid = np.linspace(0.6, 0.8, 5)
    # b_grid = np.linspace(0,1, number_of_step_in_b)
    # b_grid = [1.0]

    # c_grid = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.105, 0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18, 0.185, 0.19, 0.195, 0.200, 0.205, 0.21, 0.215, 0.22, 0.225, 0.23, 0.235, 0.24, 0.245, 0.25, 0.255, 0.26, 0.265, 0.27, 0.275, 0.28, 0.285, 0.29, 0.295, 0.3] # Reduced c_grid for faster testing

    c_grid = [0]
    
    # --- Create grids to store both winding number and band gap data ---
    winding_number_grid = np.zeros((len(b_grid), len(E)), dtype=complex)
    
    min_band_gap_grid = np.zeros((len(b_grid), len(E)))
    max_band_gap_grid = np.zeros((len(b_grid), len(E))) # New grid for max gap
    winding_number_band = 0

    # 2. Main Calculation Loop
    for c in c_grid:
        for b_idx, b in enumerate(b_grid):
            print(f"--- Starting Calculation for b = {b:.4f} ---")
            # Create tight-binding model
            coords = [[0, 0], [b, c]]
            # coords = [[0, 0], [b, c]]
            tb_model = TightBinding(coords, lambda_decay=1.0, lattice_constant=[1.0, 0], nk=100, mu=0.0, relative_Hopping=[0.1286,0.0919])
            
            # Initialize Floquet model
            data_path = f'MacBook_local_data/realistic_parameter/chiral_k={num_k:.1f}_nt_5_new/c={c:.4f}/floquet_data_Gomez_Leon_test_b={b:.2f}/'
            floquet_model = tb_model.Floquet(omegad=omega, E0=E, nt=nt, polarization=[1,1j], data_path=data_path)
            
            # Run simulation, which saves all .h5 files to disk
            energies, states = floquet_model.run(k_vals)
            
            # --- Calculate Winding Number (Vectorized) ---
            # winding_number_grid[b_idx] = floquet_model.winding_number(band_id=winding_number_band)
            # winding_number_grid[b_idx] = [min(2-a, a) for a in winding_number_grid[b_idx]]



            # floquet_model.plot_band_structure(k_vals,save_band_structure=True)
            winding_number_grid[b_idx] = floquet_model.winding_number(band_id=winding_number_band)
            print(f"Winding number calculation complete for b={b:.2f}")

            # --- Calculate Minimum and Maximum Band Gaps ---
            print("Calculating min/max band gaps by reading saved files...")
            for e_idx, E_val in enumerate(E):
                fname = os.path.join(data_path, f"band_E{E_val:.6f}.h5")
                try:
                    band_energy, _ = load_data_from_hdf5(fname)
                    if band_energy.shape[1] >= 2:
                        gaps_at_each_k = np.abs(band_energy[:, 1] - band_energy[:, 0])
                        # Store both the minimum and maximum gap
                        min_band_gap_grid[b_idx, e_idx] = np.min(gaps_at_each_k)
                        max_band_gap_grid[b_idx, e_idx] = np.max(gaps_at_each_k)
                except FileNotFoundError:
                    print(f"Warning: Data file not found for E={E_val:.6f}.")
                    min_band_gap_grid[b_idx, e_idx] = np.nan
                    max_band_gap_grid[b_idx, e_idx] = np.nan
            print("Band gap calculation complete.")
            print('')
            gc.collect()
        print("\n--- Generating Winding Number Heatmap ---")
        B, E_mesh = np.meshgrid(b_grid, E_over_omega)
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(B, E_mesh, winding_number_grid.T.real, shading='auto', cmap='viridis')
        plt.colorbar(label='Winding Number')
        plt.xlabel('Bond Length b')
        plt.ylabel(r'$E_0 / \omega$')
        plt.title(f'Floquet Winding Number Map (Band {winding_number_band})')
        plt.tight_layout()
        save_dir = f'MacBook_local_data/realistic_parameter/chiral_k={num_k:.1f}_nt_5_new/c={c:.4f}_floquet_winding_map.png'
        abs_save_dir = f'MacBook_local_data/realistic_parameter/chiral_k={num_k:.1f}_nt_5_new/absolute_band_gap_map_c={c:.4f}_.png'
        rel_save_dir = f'MacBook_local_data/realistic_parameter/chiral_k={num_k:.1f}_nt_5_new/arelative_band_gap_map_c={c:.4f}.png'
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        plt.savefig(save_dir, dpi=300)
        # plt.show()
        # 2. Absolute Band Gap Heatmap
        print("Generating Absolute Band Gap Heatmap...")
        plt.figure(figsize=(8, 6))
        # Use a logarithmic color scale to better visualize small gap values
        pcm = plt.pcolormesh(B, E_mesh, min_band_gap_grid.T, shading='auto', 
                            norm=colors.LogNorm(vmin=min_band_gap_grid[min_band_gap_grid>0].min(), vmax=min_band_gap_grid.max()),
                            cmap='magma')
        plt.colorbar(pcm, label='Minimum Gap (log scale)')
        plt.xlabel('Bond Length b')
        plt.ylabel(r'$E_0 / \omega$')
        plt.title(f'Absolute Band Gap Map (c={c:.2f})')
        plt.tight_layout()
        plt.savefig(abs_save_dir, dpi=300)
        plt.close()

        # 3. Relative Band Gap Heatmap
        print("Generating Relative Band Gap Heatmap...")
        # Calculate the relative gap, handling potential division by zero
        relative_band_gap_grid = np.divide(min_band_gap_grid, max_band_gap_grid, 
                                        out=np.zeros_like(min_band_gap_grid), 
                                        where=max_band_gap_grid!=0)
        plt.figure(figsize=(8, 6))
        pcm = plt.pcolormesh(B, E_mesh, relative_band_gap_grid.T, shading='auto', vmin=0, vmax=1, cmap='inferno')
        plt.colorbar(pcm, label='Min Gap / Max Gap')
        plt.xlabel('Bond Length b')
        plt.ylabel(r'$E_0 / \omega$')
        plt.title(f'Relative Band Gap Map (c={c:.2f})')
        plt.tight_layout()
        plt.savefig(rel_save_dir, dpi=300)
        plt.close()

        # 3. Correlated Slice Plot Generation
        if slice_b_value is not None:
            slice_idx = np.argmin(np.abs(np.array(b_grid) - slice_b_value))
            actual_b_val = b_grid[slice_idx]
            
            # Extract the 1D data for both quantities
            winding_slice_data = winding_number_grid[slice_idx]
            gap_slice_data = min_band_gap_grid[slice_idx]

            print(f"\n--- Generating Correlated Slice Plot for b = {actual_b_val:.4f} ---")
            
            # --- Create a figure with two subplots, sharing the x-axis ---
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            fig.suptitle(f'Topological Properties at Bond Length b = {actual_b_val:.4f}', fontsize=16)

            # --- Subplot 1: Winding Number ---
            ax1.plot(E_over_omega, winding_slice_data.real, 'b.-', markersize=4, label='Winding Number')
            ax1.set_ylabel('Winding Number', fontsize=12, color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_ylim(-0.1, 1.1)
            ax1.set_yticks([0, 0.5, 1.0])
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # --- Subplot 2: Band Gap ---
            ax2.plot(E_over_omega, gap_slice_data, 'r.-', markersize=4, label='Min. Band Gap')
            ax2.set_ylabel('Minimum Band Gap', fontsize=12, color='red')
            ax2.set_xlabel(r'$E_0 / \omega$', fontsize=14)
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.set_ylim(bottom=0)
            # Set y-axis to log scale and autoscale to fit all y values
            ax2.set_yscale('log')
            ax2.autoscale(enable=True, axis='y')

            # Final layout adjustments and saving
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            slice_save_dir = 'MacBook_local_data/realistic_parameter/chiral/'
            os.makedirs(slice_save_dir, exist_ok=True)
            plt.savefig(os.path.join(slice_save_dir, f'correlated_slice_c={c}_b={actual_b_val:.2f}.png'), dpi=300)
            # plt.show()
            plt.clf()
    # --- Create a single figure and axis ---
            fig, ax1 = plt.subplots(figsize=(10, 6))
            fig.suptitle(f'Topological Properties at Bond Length b = {actual_b_val:.4f}', fontsize=16)

            # --- Plot Winding Number on the first Y-axis (left) ---
            color1 = 'tab:blue'
            ax1.set_xlabel(r'$E_0 / \omega$', fontsize=14)
            ax1.set_ylabel('Winding Number', fontsize=12, color=color1)
            p1 = ax1.plot(E_over_omega, winding_slice_data.real, color=color1, linestyle='-', marker='.', markersize=4, label='Winding Number')
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.set_ylim(-0.1, 1.1)
            ax1.set_yticks([0, 0.5, 1.0])
            ax1.grid(True, linestyle='--', alpha=0.7)

            # --- Create a second Y-axis that shares the same X-axis ---
            ax2 = ax1.twinx()
            
            # --- Plot Band Gap on the second Y-axis (right) ---
            color2 = 'tab:red'
            ax2.set_ylabel('Minimum Band Gap', fontsize=12, color=color2)
            p2 = ax2.plot(E_over_omega, gap_slice_data, color=color2, linestyle='--', marker='x', markersize=4, label='Min. Band Gap')
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.set_ylim(bottom=0)

            # --- Create a combined legend ---
            lns = p1 + p2
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='upper left')

            # Final layout adjustments and saving
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            slice_save_dir = 'MacBook_local_data/realistic_parameter/chiral_k=1000.0_nt_5_new/'
            os.makedirs(slice_save_dir, exist_ok=True)
            plt.savefig(os.path.join(slice_save_dir, f'dual_axis_slice_c={c}_b={actual_b_val:.2f}.png'), dpi=300)
            # plt.show()
            plt.clf()

        # 4. Winding Number Heatmap Plot
        print("\n--- Generating Winding Number Heatmap ---")
        B, E_mesh = np.meshgrid(b_grid, E_over_omega)
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(B, E_mesh, winding_number_grid.T.real, shading='auto', cmap='viridis')
        plt.colorbar(label='Winding Number')
        plt.xlabel('Bond Length b')
        plt.ylabel(r'$E_0 / \omega$')
        plt.title(f'Floquet Winding Number Map (Band {winding_number_band})')
        plt.tight_layout()
        save_dir = f'MacBook_local_data/realistic_parameter/chiral_k=200.0_new/c={c_grid[0]:.4f}_floquet_winding_map_0_2.png'
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        plt.savefig(save_dir, dpi=300)
        # plt.show()


def test_1D_2norbs(E0 = 0.5, number_of_step_in_omega = 11, nt = 101):
    import gc
    """
    Test the Gomez-Leon 2013 model but using TightBinding and FloquetBloch classes, calculate the winding number and finally plot the heatmap
    """
    # Parameters
    # time_start = time.time()
    # omega_1 = np.linspace(0.1, 0.2, 21)
    # omega_2 = np.linspace(0.2, 1, 41)
    # omega = np.concatenate((omega_1, omega_2))
    omega = np.linspace(0.1,0.01,46)
    omega = np.linspace(0.1,0.03,36)
    omega = [0.01]
    # omega = [0.8,0.7,0.6,0.5,0.4,0.3]
    # omega = [0.24]
    # omega = [5]
    # E = np.linspace(0, 0.001, 11)
    # E_2 = np.linspace(0.001, 0.01, 10)
    # E_3 = np.linspace(0.01, 0.1, 10)
    # E_4 = np.linspace(0.1, 1, 10)
    # E = np.concatenate((E,E_2,E_3,E_4))
    # E = np.linspace(0, E0, 501)
    E = np.linspace(0, 0.2, 201)
    # E = [1.692,1.693]
    k_vals = np.linspace(-np.pi, np.pi, 100)
    winding_number_grid = np.zeros((len(omega), len(E)), dtype=complex)
    winding_number_band = 0
    for omg_idx, omg in enumerate(omega):
        # Create tight-binding model

        # tb_model = TightBinding(coords, lambda_decay=1.0, lattice_constant=[1.0], nk=100, mu=0.0, relative_Hopping=[1.5,1,1,1,1,1])
        # coords = [[0], [b],[0.6],[0.8]]
        # tb_model = TightBinding(coords, lambda_decay=1.0, lattice_constant=[1.0], nk=100, mu=0.0, relative_Hopping=[1.5,1,1,1,1,1,1,1,1,1,1,1])   
        coords = [[0], [0.7]]   
        tb_model = TightBinding(coords, lambda_decay=1.0, lattice_constant=[1.0], nk=200, mu=0.0, relative_Hopping=[0.1286,0.0919])      
        # Run Floquet analysis
        floquet_model = tb_model.Floquet(omegad=omg, E0=E, nt=nt, polarization=[1], data_path=f'MacBook_local_data/E_omega_b=0.7_realistic_parameter_E_max_1_k=6/floquet_data_1D_2norbs_test_omega={omg:.5f}/')
        energies, states = floquet_model.run(k_vals)

        winding_number_grid[omg_idx]=floquet_model.winding_number(band_id=0)
        print(f"Winding number for omega={omg:.4f}: {winding_number_grid[omg_idx]}")
        
        
        # floquet_model.plot_band_structure(k_vals,save_band_structure=True)
        gc.collect()
        print('')
    # Convert b_grid and E to 2D meshgrid for plotting
    B, E_mesh = np.meshgrid(omega, E)

    # Plot the winding number map (real part only if complex)
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(B, E_mesh, winding_number_grid.T.real, shading='auto', cmap='viridis')
    plt.colorbar(label='Winding Number')
    plt.xlabel('Driving Frequency omega')
    plt.ylabel(r'$E_0$ (Electric Field Amplitude)')
    plt.title(f'Floquet Winding Number Map (Band {winding_number_band}/)')
    plt.tight_layout()
    plt.show()

def test_Gomez_Leon_2013_test_only(E0=20, number_of_step_in_b=11, nt=11, slice_b_value=None):
    """
    Calculates winding number and band gap, then plots a correlated slice
    plot and the final winding number heatmap.
    """
    np.set_printoptions(threshold=np.inf)
    # 1. Parameters Setup
    omega = 1
    # E_over_omega = np.linspace(8, 10, 101) # Reduced steps for faster testing
    E_over_omega = np.linspace(0, E0/omega, 201) # Reduced steps for faster testing
    E = [e * omega for e in E_over_omega]
    num_k = 200  # Reduced k values for faster testing
    # k_vals = np.linspace(-np.pi, np.pi, num_k)
    k_vals = np.stack([np.linspace(-np.pi, np.pi, num_k), np.zeros(num_k)], axis=1)
    # k_vals = np.stack([np.linspace(-2.9, -2.8, 10), np.zeros(10)], axis=1)
    # k_vals = np.stack([np.linspace(-1.8, -1.6, 10), np.zeros(10)], axis=1)
    # k_vals = np.stack([np.linspace(-np.pi, -np.pi+0.1, 10), np.zeros(10)], axis=1)
    # k_vals = np.stack([np.linspace(-0.1, 0.1, 30), np.zeros(30)], axis=1)
    # b_grid = np.linspace(0, 1, number_of_step_in_b)
    # b_grid = np.linspace(1, 0, number_of_step_in_b)
    b_grid = [0.6]

    # c_grid = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.105, 0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18, 0.185, 0.19, 0.195, 0.200, 0.205, 0.21, 0.215, 0.22, 0.225, 0.23, 0.235, 0.24, 0.245, 0.25, 0.255, 0.26, 0.265, 0.27, 0.275, 0.28, 0.285, 0.29, 0.295, 0.3] # Reduced c_grid for faster testing


    # 2. Main Calculation Loop
    for External_E in E:
        print(f"--- Starting Calculation for E = {External_E:.6f} ---")
        for b_idx, b in enumerate(b_grid):
            os.makedirs(f'MacBook_local_data/for overlap matrix/no_reassign_test_new_test_only/chiral_k={num_k:.1f}_nt_5_new/c=0/floquet_data_Gomez_Leon_test_b={b:.2f}/overlap_matrix', exist_ok=True)
            os.makedirs(f'MacBook_local_data/for overlap matrix/no_reassign_test_new_test_only/chiral_k={num_k:.1f}_nt_5_new/c=0/floquet_data_Gomez_Leon_test_b={b:.2f}/overlap_matrix_valance_band', exist_ok=True)
            os.makedirs(f'MacBook_local_data/for overlap matrix/no_reassign_test_new_test_only/chiral_k={num_k:.1f}_nt_5_new/c=0/floquet_data_Gomez_Leon_test_b={b:.2f}/overlap_matrix_conduction_band', exist_ok=True)
            print(f"--- Starting Calculation for b = {b:.4f} ---")
            # Create tight-binding model
            coords = [[0, 0], [b, 0]]
            # coords = [[0, 0], [b, c]]
            tb_model = TightBinding(coords, lambda_decay=1.0, lattice_constant=[1.0, 0], nk=100, mu=0.0, relative_Hopping=[0.1286,0.0919])
            
            # Initialize Floquet model
            data_path = f'MacBook_local_data/for overlap matrix/no_reassign_test_new_test_only/chiral_k={num_k:.1f}_nt_5_new/c=0/floquet_data_Gomez_Leon_test_b={b:.2f}/'
            floquet_model = tb_model.Floquet(omegad=omega, E0=E, nt=nt, polarization=[1,1j], data_path=data_path)
            # floquet_model.run(k_vals)
            # print(floquet_model.winding_number(band_id=0))
            # floquet_model.plot_band_structure(k_vals,save_band_structure=True)
            # time.sleep(10)
            # Run simulation, which saves all .h5 files to disk
            energies, states = floquet_model.track_Floquet_Modes(k_vals, External_E, omega)
            print(f'overlap_matrix_for_different_k_at_E = {External_E:.6f}')
            # overlap_matrix = np.matmul(states[:,:,2], states[:,:,3].conj().T)
            overlap_matrix = np.matmul(states[:,:,0], states[:,:,1].conj().T)
            # print(overlap_matrix.shape)
            overlap_matrix = np.round(overlap_matrix, 6)
            #check if there are non-zero elements in the overlap matrix till 6 decimal places
            if np.count_nonzero(overlap_matrix) != 0:
                pass
                # print(f"found non-zero elements in the overlap matrix for E = {External_E:.6f} and b = {b:.2f}")
                # #print which elements are non-zero
                # non_zero_elements = np.argwhere(overlap_matrix != 0)
                # print(f"Non-zero elements in the overlap matrix for E = {External_E:.6f} and b = {b:.2f}:")
                # for elem in non_zero_elements:
                #     print(f"Element at {elem} = {overlap_matrix[elem[0], elem[1]]:.6f}")
                # time.sleep(3)
            #save the overlap matrix to disk
            overlap_matrix_path = os.path.join(data_path, f'overlap_matrix/overlap_matrix_E={External_E:.6f}_b={b:.2f}.txt')
            np.savetxt(overlap_matrix_path, overlap_matrix, fmt='%.6f')


            overlap_matrix = np.matmul(states[:,:,0], states[:,:,0].conj().T)
            # overlap_matrix = np.matmul(states[:,:,2], states[:,:,2].conj().T)
            # print(overlap_matrix.shape)
            overlap_matrix = np.round(overlap_matrix, 6)
            #check if there are non-zero elements in the overlap matrix till 6 decimal places
            if np.count_nonzero(overlap_matrix) != 0:
                # print(f"found non-zero elements in the overlap matrix for E = {External_E:.6f} and b = {b:.2f}")
                #print which elements are non-zero
                non_zero_elements = np.argwhere(overlap_matrix != 0)
                # print(f"Non-zero elements in the overlap matrix for E = {External_E:.6f} and b = {b:.2f}:")
                for elem in non_zero_elements:
                    pass
                    # print(f"Element at {elem} = {overlap_matrix[elem[0], elem[1]]:.6f}")
                # time.sleep(3)
            #save the overlap matrix to disk
            overlap_matrix_path = os.path.join(data_path, f'overlap_matrix_valance_band/overlap_matrix_E={External_E:.6f}_b={b:.2f}.txt')
            np.savetxt(overlap_matrix_path, overlap_matrix, fmt='%.6f')
            overlap_matrix = np.matmul(states[:,:,1], states[:,:,1].conj().T)
            # overlap_matrix = np.matmul(states[:,:,3], states[:,:,3].conj().T)
            print(overlap_matrix.shape)
            overlap_matrix = np.round(overlap_matrix, 6)
            #check if there are non-zero elements in the overlap matrix till 6 decimal places
            if np.count_nonzero(overlap_matrix) != 0:
                pass
                # print(f"found non-zero elements in the overlap matrix for E = {External_E:.6f} and b = {b:.2f}")
                # #print which elements are non-zero
                # non_zero_elements = np.argwhere(overlap_matrix != 0)
                # print(f"Non-zero elements in the overlap matrix for E = {External_E:.6f} and b = {b:.2f}:")
                # for elem in non_zero_elements:
                #     print(f"Element at {elem} = {overlap_matrix[elem[0], elem[1]]:.6f}")
                # time.sleep(3)
            #save the overlap matrix to disk
            overlap_matrix_path = os.path.join(data_path, f'overlap_matrix_conduction_band/overlap_matrix_E={External_E:.6f}_b={b:.2f}.txt')
            np.savetxt(overlap_matrix_path, overlap_matrix, fmt='%.6f')


            # print(overlap_matrix)
            # fig, ax = plt.subplots()
            # colors = ['red', 'blue']
            # # ax.plot(k_vals, energies.real)
            # # ax.plot(k_vals, energies.real+omega)
            # # ax.plot(k_vals, energies.real-omega)
            # # ax.plot(k_vals, energies.real+2*omega)
            # # ax.plot(k_vals, energies.real-2*omega)
            # # ax.plot(k_vals, energies.real+3*omega)
            # # ax.plot(k_vals, energies.real-3*omega)
            # # ax.plot(k_vals, energies.real+4*omega)
            # # ax.plot(k_vals, energies.real-4*omega)
            # k_vals_plot = k_vals[:, 0]  # Use only the first column for 1D k-values
            # for band_idx in range(2):
            #     ax.plot(k_vals_plot, energies[:, band_idx].real,
            #             label=f"band {band_idx}", linestyle='dashed', color=colors[band_idx])
            #     ax.plot(k_vals_plot, energies[:, band_idx].real+omega, color=colors[band_idx])
            #     ax.plot(k_vals_plot, energies[:, band_idx].real-omega, color=colors[band_idx])
            #     ax.plot(k_vals_plot, energies[:, band_idx].real+2*omega, color=colors[band_idx])
            #     ax.plot(k_vals_plot, energies[:, band_idx].real-2*omega, color=colors[band_idx])
            #     ax.plot(k_vals_plot, energies[:, band_idx].real+3*omega, color=colors[band_idx])
            #     ax.plot(k_vals_plot, energies[:, band_idx].real-3*omega, color=colors[band_idx])
            #     ax.plot(k_vals_plot, energies[:, band_idx].real+4*omega, color=colors[band_idx])
            #     ax.plot(k_vals_plot, energies[:, band_idx].real-4*omega, color=colors[band_idx])
            # ax.set_xlabel("k")
            # ax.set_ylabel("Energy")
            # ax.set_title(f"Floquet bands at E = {External_E:.6f}")
            # ax.legend()
            # target = os.path.join(data_path, "Floquet_Band_Structure")
            # os.makedirs(target, exist_ok=True)
            # out_png = os.path.join(target, f"band_E{External_E:.6f}.png")
            # fig.savefig(out_png)
            # plt.close(fig)
            # print(f"Saved plot to {out_png}")
        gc.collect()
        # time.sleep(1)


def analyze_and_plot_overlap_across_E(k_fixed, E_values, k_values, data_path):
    """
    Post-processes saved Floquet data to calculate and plot the overlap matrix
    for a fixed k-point across a range of E values.

    This function should be called AFTER a full simulation (like test_Gomez_Leon_2013_test_only)
    has been run, which generates all the necessary band_E*.h5 files.

    Args:
        k_fixed (float): The fixed momentum value you want to analyze.
        E_values (list or np.ndarray): The full list of E-field values used in the simulation.
        k_values (np.ndarray): The array of k-points used in the simulation (shape [num_k, dim]).
        data_path (str): The path to the directory where the .h5 files are stored.
    """
    print("\n--- Starting Post-Processing Analysis ---")
    print(f"Analyzing overlap matrix at fixed k = {k_fixed:.4f}")

    # 1. Find the index corresponding to the fixed k-point
    # We find the k-point in your grid that is closest to the desired k_fixed value.
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
            # load_data_from_hdf5 returns (energies, states)
            # states is a list of arrays: [ground_band_states, excited_band_states, ...]
            _, all_bands_states = load_data_from_hdf5(fname)

            # Extract the ground state vector (band 0) at our specific k-point index
            ground_state_vec = all_bands_states[0][k_idx, :]
            ground_states_at_k.append(ground_state_vec)

            # Extract the excited state vector (band 1) at the same k-point
            excited_state_vec = all_bands_states[1][k_idx, :]
            excited_states_at_k.append(excited_state_vec)

        except FileNotFoundError:
            print(f"Warning: Data file not found for E={E_val:.6f}. Skipping this point.")
            continue
        except Exception as e:
            print(f"An error occurred while loading data for E={E_val:.6f}: {e}")
            continue
            
    if not ground_states_at_k:
        print("Error: No data was loaded. Please check the data_path and E_values.")
        return

    # 3. Convert lists to NumPy arrays and calculate the overlap matrix
    ground_states = np.array(ground_states_at_k)  # Shape: (num_E, state_dimension)
    excited_states = np.array(excited_states_at_k) # Shape: (num_E, state_dimension)

    # Calculate M_ij = <ground(E_i) | excited(E_j)>
    overlap_matrix = np.matmul(ground_states, excited_states.conj().T)
    print(f"Successfully computed overlap matrix of shape {overlap_matrix.shape}")

    # 4. Generate the graphical representation (heatmap)
    plt.figure(figsize=(8, 7))
    # We plot the absolute value of the overlap matrix
    im = plt.imshow(np.abs(overlap_matrix), origin='upper', aspect='equal', cmap='viridis', interpolation='nearest')
    
    plt.title(f"Overlap Matrix Heatmap at k = {actual_k:.4f}", fontsize=14)
    plt.xlabel("E-point index (j)", fontsize=12)
    plt.ylabel("E-point index (i)", fontsize=12)
    
    cbar = plt.colorbar(im)
    cbar.set_label(r'$|<\psi_g(E_i)|\psi_e(E_j)>|$', fontsize=12)
    
    # Save the plot
    plot_dir = os.path.join(data_path, "overlap_plots_fixed_k")
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, f"overlap_heatmap_k={actual_k:.4f}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved overlap heatmap to {save_path}")
    plt.show()
    
if __name__ == "__main__":
    # --- Step 1: Run your existing simulation to generate the data ---
    # (I'm using the parameters from your test_Gomez_Leon_2013_test_only function)
    
    print("--- STEP 1: RUNNING FULL SIMULATION ---")
    omega = 1
    E0 = 20
    E_over_omega = np.linspace(0, E0/omega, 201)
    E = [e * omega for e in E_over_omega]
    num_k = 200
    k_vals = np.stack([np.linspace(-np.pi, np.pi, num_k), np.zeros(num_k)], axis=1)
    b_grid = [0.6]
    b = b_grid[0] # The b value we used
    
    # The main data path used by the simulation
    data_path = f'MacBook_local_data/for overlap matrix/no_reassign_test_new_test_only/chiral_k={num_k:.1f}_nt_5_new/c=0/floquet_data_Gomez_Leon_test_b={b:.2f}/'

    # This function is your original function that runs the simulation and saves .h5 files
    # Make sure it runs and completes successfully before the next step.
    test_Gomez_Leon_2013() 

    # --- Step 2: Run the new analysis function on the saved data ---
    
    print("\n--- STEP 2: POST-PROCESSING FOR OVERLAP MATRIX ---")
    
    # Define the fixed k-point you want to investigate from your phase diagram
    k_to_analyze = 0.0 # Example: The Gamma point (k=0)

    analyze_and_plot_overlap_across_E(
        k_fixed=k_to_analyze,
        E_values=E,
        k_values=k_vals,
        data_path=data_path
    )