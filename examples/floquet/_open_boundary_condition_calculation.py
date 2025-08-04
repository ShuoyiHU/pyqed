import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import jv
from scipy.linalg import eigh
from pyqed.floquet.floquet_to_be_fixed import TightBinding, FloquetBloch
import gc

def main(E_min = 0, E_max=5, E_step =201, omega=0.5, OBC_cells=20, nt = 101):
    """
    Main function to perform a sweep over a list of E-field values,
    analyze the Floquet system for each, and save all plots and data
    into organized subdirectories.
    """
    print("--- Starting Floquet OBC Analysis for Multiple E-Fields ---")

    # --- 1. System and Path Parameters ---
    
    # Define system parameters
    N0 = nt // 2
    all_p = np.arange(-N0, N0 + 1)
    
    # Define the list of E-fields to test
    E_field_list = np.linspace(E_min, E_max, E_step) 
    print(f"Will analyze the system for E-fields: {E_field_list}\n")

    # --- 2. Define the SSH Model (once) ---
    model = TightBinding(
        coords=[[0], [0.75]],
        relative_Hopping=[0.1286,0.0919], # t1 < t2 for topological phase without E-field
        # relative_Hopping=[1.5, 1], # t1 < t2 for topological phase without E-field
        lattice_constant=[1.0]
    )
    norbs = model.norb
    # Set the fixed data path as requested
    base_data_path = f"MacBook_local_data/open_boundary_condition_calculation/coords_{model.coords[0][0]}_{model.coords[1][0]}_t1_{model.relative_Hopping[0]}_t2_{model.relative_Hopping[1]}/floquet_data_Gomez_Leon_test_b=0.60/E_max_{E_max}_omega_{omega}_OBC_cells_{OBC_cells}_nt_{nt}"
    # Create paths for organized output
    spectrum_plots_path = os.path.join(base_data_path, "1_Full_Spectrum_Plots")
    edge_state_plots_path = os.path.join(base_data_path, "2_Edge_State_Plots")
    os.makedirs(spectrum_plots_path, exist_ok=True)
    os.makedirs(edge_state_plots_path, exist_ok=True)
    # --- Loop over each E-field value ---
    for E_field in E_field_list:
        print(f"--- Processing E = {E_field:.3f} ---")
        
        # --- 3. Build the Floquet Hamiltonian for the current E-field ---
        print("Building Floquet OBC Hamiltonian...")
        H_intra_static = model.intra
        H_inter_static = model.inter_upper
        d_intra = model.coords[1] - model.coords[0]
        d_inter = (model.coords[0] + model.lattice_constant) - model.coords[1]
        H_intra_p, H_inter_p = {}, {}
        Z_intra, Z_inter = (E_field / omega) * d_intra[0], (E_field / omega) * d_inter[0]
        for p in all_p:
            H_intra_p[p] = H_intra_static * jv(p, Z_intra)
            H_inter_p[p] = H_inter_static * jv(p, Z_inter)
        N_obc_static = OBC_cells * norbs
        N_total = N_obc_static * nt
        F_OBC = np.zeros((N_total, N_total), dtype=complex)
        for n in range(nt):
            for m in range(nt):
                p = n - m
                H_p_intra = H_intra_p.get(p, np.zeros_like(H_intra_static))
                H_p_inter = H_inter_p.get(p, np.zeros_like(H_inter_static))
                H_p_OBC = np.kron(np.eye(OBC_cells), H_p_intra) + \
                          np.kron(np.diag(np.ones(OBC_cells - 1), 1), H_p_inter) + \
                          np.kron(np.diag(np.ones(OBC_cells - 1), -1), H_p_inter.conj().T)
                if n == m:
                    H_p_OBC += np.eye(N_obc_static) * (n - N0) * omega
                row_start, row_end = n * N_obc_static, (n + 1) * N_obc_static
                col_start, col_end = m * N_obc_static, (m + 1) * N_obc_static
                F_OBC[row_start:row_end, col_start:col_end] = H_p_OBC

        # --- 4. Diagonalize and Analyze ---
        print(f"Diagonalizing {F_OBC.shape[0]}x{F_OBC.shape[0]} matrix...")
        eigvals, eigvecs = eigh(F_OBC)
        print(eigvecs.shape)
        # --- 5. Generate and Save Full Spectrum Plot ---
        mean_positions = []
        for i in range(len(eigvals)):
            state_vec = eigvecs[:, i]
            state_reshaped = state_vec.reshape((nt, OBC_cells, norbs))
            prob_dist = np.sum(np.abs(state_reshaped)**2, axis=(0, 2))
            mean_pos = np.sum(np.arange(OBC_cells) * prob_dist)
            mean_positions.append(mean_pos)
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(
            np.arange(len(eigvals)), eigvals.real, c=mean_positions,
            cmap='coolwarm', s=15, vmin=0, vmax=OBC_cells-1, alpha=0.9,
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Mean Position in Chain (<x>)', fontsize=12)
        ax.set_xlabel("State Index", fontsize=12)
        ax.set_ylabel("Quasi-energy", fontsize=12)
        ax.set_title(f"Floquet Spectrum at E = {E_field:.3f} (N={OBC_cells})", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(-omega / 2, omega / 2)
        ax.set_xlim(OBC_cells * norbs * (nt/2-0.5), OBC_cells * norbs * (nt/2+0.5))
        # ax.set_ylim(-3* omega / 2,3* omega / 2)
        
        plot_filename = os.path.join(spectrum_plots_path, f"spectrum_E_{E_field:.3f}.png")
        plt.savefig(plot_filename, dpi=300)
        print(f"  -> Saved full spectrum plot to: {plot_filename}")
        plt.close(fig)

        # --- 6. Isolate, Save, and Plot the Two Edge States ---
        edge_state_indices = np.argsort(np.abs(eigvals.real))[:2]
        
        for i, state_idx in enumerate(edge_state_indices):
            edge_state_vec = eigvecs[:, state_idx]
            edge_state_energy = eigvals[state_idx].real
            prob_dist = np.sum(np.abs(edge_state_vec.reshape((nt, OBC_cells, norbs)))**2, axis=(0, 2))
            
            dist_filename = os.path.join(edge_state_plots_path, f"edge_state_{i+1}_E_{E_field:.3f}_distribution.txt")
            np.savetxt(dist_filename, np.vstack((np.arange(OBC_cells), prob_dist)).T,
                       header=f"Site_Index Probability (Energy={edge_state_energy:.6f})", fmt='%d %.8e')

            fig_edge, ax_edge = plt.subplots(figsize=(10, 6))
            ax_edge.bar(np.arange(OBC_cells), prob_dist, width=0.8, alpha=0.7,
                        label=f'Quasi-energy = {edge_state_energy:.4f}')
            ax_edge.plot(np.arange(OBC_cells), prob_dist, 'o-', color='crimson', markersize=4)
            ax_edge.set_xlabel("Unit Cell Index", fontsize=12)
            ax_edge.set_ylabel("Probability Density |ψ(x)|²", fontsize=12)
            ax_edge.set_title(f"Distribution of Edge State {i+1} at E = {E_field:.3f}", fontsize=14)
            ax_edge.legend()
            ax_edge.set_xlim(-1, OBC_cells)

            plot_filename_edge = os.path.join(edge_state_plots_path, f"edge_state_{i+1}_E_{E_field:.3f}_plot.png")
            plt.savefig(plot_filename_edge, dpi=300)
            plt.close(fig_edge)
        gc.collect()  # Clear memory after each E-field analysis
        print(f"  -> Saved individual edge state data and plots.\n")

if __name__ == "__main__":
    # main(omega=0.2)
    # main(omega=0.3)
    main(E_min = 0.014, E_max=0.037, E_step=24, omega=0.0025)
    main(E_max=0.05, E_step=51, omega=0.002, nt=201)