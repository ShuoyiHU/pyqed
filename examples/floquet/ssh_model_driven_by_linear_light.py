import numpy as np
import matplotlib.pyplot as plt
from  pyqed.floquet.floquet import TightBinding, FloquetBloch
from dataclasses import dataclass, field
from typing import List, Tuple

# This example calculates and plots the Floquet winding number map for a 1D SSH model using external field strength and frequency as parameters, defult is driven by linear light, the model belongs to BDI class, winding number is derived from Zak Phase.
# jump to if __name__ == "__main__" to adjust the system parameters and run the script directly

@dataclass
class ModelParameters:
    """Stores all parameters related to the tight-binding model itself."""
    coords: List[List[float]]
    relative_Hopping: List[float]
    lambda_decay: float = 1.0
    lattice_constant: float = 1.0
    mu: float = 0.0 # on site potential, default is zero
    # Note: nk (number of k-points for model) is handled by FieldParameters
    # to ensure consistency with the k-space scan.

@dataclass
class FieldParameters:
    """Stores all parameters for the external field and simulation scan."""
    E0_max: float = 100.0
    num_steps_E: int = 101
    omega_range: Tuple[float, float] = (8.0, 10.0)
    num_steps_omega: int = 11
    nt: int = 11 # number of Fourier components in time domain
    num_k_points: int = 100   # For k_vals resolution and nk in TB model
    polarization: List[int] = field(default_factory=lambda: [1])
    target_band_id: int = 0
    base_data_path: str = 'my_favorite_data_path/E_omega_high_frequency_test/'
    save_band_structure: bool = True


# main function used for winding number calculation and plotting

def calculate_and_plot_winding_map(model_params: ModelParameters, field_params: FieldParameters):
    """
    Calculates and plots the Floquet winding number map using structured configuration.

    Args:
        model_params (ModelParameters): An object containing the tight-binding model parameters.
        field_params (FieldParameters): An object containing the external field and simulation parameters.
    """
    # --- 1. Setup parameters ---
    omega_vals = np.linspace(*field_params.omega_range, field_params.num_steps_omega)
    E_vals = np.linspace(0, field_params.E0_max, field_params.num_steps_E)
    k_vals = np.linspace(-np.pi, np.pi, field_params.num_k_points)

    winding_number_grid = np.zeros((len(omega_vals), len(E_vals)), dtype=float)

    print("--- Starting Calculation ---")
    # --- 2. Main calculation loop ---
    for omg_idx, omg in enumerate(omega_vals):
        # Create tight-binding model using parameters from the config object
        tb_model = TightBinding(
            coords=model_params.coords,
            lambda_decay=model_params.lambda_decay,
            lattice_constant=model_params.lattice_constant,
            nk=field_params.num_k_points,  # Ensure consistency
            mu=model_params.mu,
            relative_Hopping=model_params.relative_Hopping
        )
        
        # Define a specific data path for each omega value
        data_path = f'{field_params.base_data_path}floquet_data_1D_2norbs_test_omega={omg:.5f}/'

        # Run Floquet analysis
        floquet_model = tb_model.Floquet(
            omegad=omg,
            E0=E_vals,
            nt=field_params.nt,
            polarization=field_params.polarization,
            data_path=data_path
        )
        
        floquet_model.run(k_vals)
        winding_number_grid[omg_idx, :] = floquet_model.winding_number(band_id=field_params.target_band_id)
        
        print(f"Calculation complete for omega={omg:.2f}")
        
        # Optional: Plot band structure for each external field condition for examination of the band structure
        if field_params.save_band_structure:
            floquet_model.plot_band_structure(k_vals, save_band_structure=True)
        
    # --- 3. Plotting the final winding nubmer heatmap ---
    print("--- Plotting Results ---")
    omega_mesh, E_mesh = np.meshgrid(omega_vals, E_vals)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(omega_mesh, E_mesh, winding_number_grid.T, shading='auto', cmap='viridis')
    plt.colorbar(label='Winding Number')
    plt.xlabel('Driving Frequency (ω)')
    plt.ylabel(r'Electric Field Amplitude ($E_0$)')
    plt.title(f'Floquet Winding Number Map (Band {field_params.target_band_id})')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Define the parameters for the physical model
    gomez_leon_model = ModelParameters(
        coords=[[0], [0.7]],
        relative_Hopping=[1.5, 1.0]
    )
    # Define the parameters for the simulation and external field
    simulation_setup = FieldParameters(
        E0_max=100.0,
        num_steps_E=101,
        omega_range=(8.0, 10.0),
        num_steps_omega=11,
        nt=11, # Number of Fourier components in time domain, set larger if result does not converge
        save_band_structure=False,  # Set to True if you want to save individual band structures
        base_data_path='my_favorite_data_path/E_omega_high_frequency_test/'  # Adjust as needed
    )

    # run winding number calculation and plotting
    calculate_and_plot_winding_map(
        model_params=gomez_leon_model,
        field_params=simulation_setup
    )

