import numpy as np
import matplotlib.pyplot as plt
from  pyqed.floquet.floquet import TightBinding, FloquetBloch
import os
from pyqed.floquet.utils import track_valence_band, berry_phase_winding, figure, track_valence_band_GL2013, save_data_to_hdf5, load_data_from_hdf5
import matplotlib.colors as colors
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# Step 1: Define the configuration classes

@dataclass
class GomezLeonModelParameters:
    """Stores parameters for the Gomez-Leon tight-binding model."""
    b_values: List[float] = field(default_factory=lambda: [0.75])
    c_values: List[float] = field(default_factory=lambda: [0.0])
    relative_Hopping: List[float] = field(default_factory=lambda: [1.5, 1.0])
    lambda_decay: float = 1.0
    lattice_constant: List[float] = field(default_factory=lambda: [1.0, 0.0])
    mu: float = 0.0

@dataclass
class GomezLeonFieldParameters:
    """Stores parameters for the field, simulation, and analysis."""
    omega: float = 10.0
    E0_max: float = 200.0
    num_steps_E: int = 101
    num_k_points: int = 500
    nt: int = 11  # 'nt'
    polarization: List[complex] = field(default_factory=lambda: [1, 1j])
    target_band_id: int = 0
    slice_b_value: Optional[float] = None # Optional value of 'b' for slice plots
    base_data_path: str = 'my_favorite_data_path/refactored_gomez_leon/'


# Step 2: Create the refactored main function

def run_gomez_leon_simulation(model_params: GomezLeonModelParameters, field_params: GomezLeonFieldParameters):
    """
    Calculates and plots winding number and band gaps for the Gomez-Leon model.
    """
    # --- 1. Setup simulation space ---
    E_over_omega = np.linspace(0, field_params.E0_max / field_params.omega, field_params.num_steps_E)
    E_vals = E_over_omega * field_params.omega
    k_vals = np.stack([np.linspace(-np.pi, np.pi, field_params.num_k_points), np.zeros(field_params.num_k_points)], axis=1)

    # --- 2. Main calculation loop over model geometry ---
    for c in model_params.c_values:
        b_grid = model_params.b_values
        
        # Initialize grids to store results for this 'c' value
        winding_number_grid = np.zeros((len(b_grid), len(E_vals)), dtype=float)
        min_band_gap_grid = np.zeros((len(b_grid), len(E_vals)))
        
        for b_idx, b in enumerate(b_grid):
            print(f"--- Starting Calculation for c = {c:.4f}, b = {b:.4f} ---")
            
            # Create tight-binding model
            coords = [[0, 0], [b, c]]
            tb_model = TightBinding(
                coords=coords,
                lambda_decay=model_params.lambda_decay,
                lattice_constant=model_params.lattice_constant,
                nk=field_params.num_k_points,
                mu=model_params.mu,
                relative_Hopping=model_params.relative_Hopping
            )
            
            # Initialize and run Floquet model
            data_path = f'{field_params.base_data_path}c={c:.4f}/b={b:.4f}/'
            floquet_model = tb_model.Floquet(
                omegad=field_params.omega,
                E0=E_vals,
                nt=field_params.nt,
                polarization=field_params.polarization,
                data_path=data_path
            )
            floquet_model.run(k_vals)
            
            # Calculate winding number and band gaps
            winding_number_grid[b_idx, :] = floquet_model.winding_number(band_id=field_params.target_band_id)
            
            for e_idx, E_val in enumerate(E_vals):
                fname = os.path.join(data_path, f"band_E{E_val:.6f}.h5")
                try:
                    band_energy, _ = load_data_from_hdf5(fname)
                    gaps = np.abs(band_energy[:, 1] - band_energy[:, 0])
                    min_band_gap_grid[b_idx, e_idx] = np.min(gaps) if gaps.size > 0 else np.nan
                except (FileNotFoundError, IOError):
                    min_band_gap_grid[b_idx, e_idx] = np.nan

            print(f"Calculation complete for b={b:.4f}")
            print("-" * 20)

        # --- 3. Plotting results for the current 'c' value ---
        plot_results_for_c(c, b_grid, E_over_omega, winding_number_grid, min_band_gap_grid, field_params)


def plot_results_for_c(c, b_grid, E_over_omega, winding_grid, gap_grid, field_params):
    """Helper function to generate all plots for a given 'c' value."""
    print(f"\n--- Generating Plots for c = {c:.4f} ---")
    
    # --- Winding Number Heatmap ---
    B_mesh, E_mesh = np.meshgrid(b_grid, E_over_omega)
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(B_mesh, E_mesh, winding_grid.T, shading='auto', cmap='viridis')
    plt.colorbar(label='Winding Number')
    plt.xlabel('Bond Length b')
    plt.ylabel(r'$E_0 / \omega$')
    plt.title(f'Floquet Winding Number Map (c={c:.4f}, Band {field_params.target_band_id})')
    plt.tight_layout()
    save_dir = os.path.join(field_params.base_data_path, f'c={c:.4f}')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'winding_map.png'), dpi=300)
    plt.close()

    # --- Band Gap Heatmap ---
    plt.figure(figsize=(8, 6))
    valid_gaps = gap_grid[gap_grid > 0]
    norm = colors.LogNorm(vmin=valid_gaps.min(), vmax=valid_gaps.max()) if valid_gaps.size > 0 else None
    plt.pcolormesh(B_mesh, E_mesh, gap_grid.T, shading='auto', cmap='magma', norm=norm)
    plt.colorbar(label='Minimum Gap (log scale)')
    plt.xlabel('Bond Length b')
    plt.ylabel(r'$E_0 / \omega$')
    plt.title(f'Minimum Band Gap Map (c={c:.4f})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'band_gap_map.png'), dpi=300)
    plt.close()

    # --- Correlated Slice Plot (if requested) ---
    if field_params.slice_b_value is not None:
        slice_idx = np.argmin(np.abs(np.array(b_grid) - field_params.slice_b_value))
        actual_b_val = b_grid[slice_idx]
        
        winding_slice = winding_grid[slice_idx]
        gap_slice = gap_grid[slice_idx]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        fig.suptitle(f'Topological Properties at c={c:.4f}, b≈{actual_b_val:.4f}', fontsize=16)

        color1 = 'tab:blue'
        ax1.set_xlabel(r'$E_0 / \omega$', fontsize=14)
        ax1.set_ylabel('Winding Number', fontsize=12, color=color1)
        p1 = ax1.plot(E_over_omega, winding_slice, color=color1, marker='.', label='Winding Number')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Minimum Band Gap (log scale)', fontsize=12, color=color2)
        p2 = ax2.plot(E_over_omega, gap_slice, color=color2, linestyle='--', marker='x', label='Min. Band Gap')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_yscale('log')

        lns = p1 + p2
        ax1.legend(lns, [l.get_label() for l in lns], loc='upper left')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f'correlated_slice_b={actual_b_val:.2f}.png'), dpi=300)
        plt.close()


# Step 3: Use the `main` block to configure and run

if __name__ == "__main__":
    # --- Configuration Area ---
    # Define the physical model parameters to scan
    model_config = GomezLeonModelParameters(
        b_values=np.linspace(0, 1, 11),  # Scan over a range of 'b'
        # c_values=[0.0, 0.1]              # Run the full simulation for each 'c'
        c_values=[0.0]              # Run the full simulation for each 'c'
    )

    # Define the simulation and field parameters
    field_config = GomezLeonFieldParameters(
        E0_max=200.0,
        num_steps_E=101,
        nt=11,
        slice_b_value=0.75, # Request a slice plot for band_gap and winding number relationship at b=0.75
        base_data_path='my_favorite_data_path/refactored_gomez_leon/'  # Adjust as needed
    )

    # --- Execution Area ---
    run_gomez_leon_simulation(
        model_params=model_config,
        field_params=field_config
    )
