import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime

# Import the necessary functions from your provided files
# Assumes the files are in the same directory or Python path
try:
    from pyqed.qchem.dvr.hybrid_gauss_dvr_method_add_d_orbitals import (
        Molecule,
        build_method2,
        scf_rhf_method2,
        save_scf_snapshot
    )
    # Import specific exponents if needed, or define locally
    # We define locally to ensure self-containment if imports fail
except ImportError:
    print("Error: Could not import 'hybrid_gauss_dvr_method_add_d_orbitals.py'.")
    print("Please ensure the file is in the current directory.")
    raise

# =============================================================================
# CONFIGURATION
# =============================================================================

# 1. Run Parameters
NZ_VAL      = 31       # The Nz value to test (increase this to check oscillation reduction)
M_VAL       = 6         # Fixed M (number of contracted Gaussians per slice)
LZ_VAL      = 6.0       # Box size (-Lz to +Lz)
NELEC       = 2          # H2 has 2 electrons

# 2. Basis Set (H 6-31G s-type contraction for example, or STO-6G)
# Using the 6-31G s-shell exponents for H as seen in your reference code
H_S_EXPS = np.array([35.52322122,
6.513143725,
1.822142904,
0.6259552659,
0.2430767471,
0.1001124280], dtype=float)
H_P_EXPS = np.array([], dtype=float) # No p-orbitals for now
H_D_EXPS = np.array([], dtype=float) # No d-orbitals for now


R_START     = 0.8
R_END       = 4.0
NUM_POINTS  = 50  # High resolution to see oscillations
BOND_LENGTHS = np.linspace(R_START, R_END, NUM_POINTS)

# Output folder
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"results_H2_PES_Nz{NZ_VAL}_M{M_VAL}_{TIMESTAMP}"

#  =============================================================================
# MAIN SCRIPT
# =============================================================================

def run_pes_scan():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print(f"=== H2 PES Scan (Pure SCF, No Newton) ===")
    print(f"Configuration: Nz={NZ_VAL}, M={M_VAL}, Lz={LZ_VAL}")
    print(f"Basis: s-exps={H_S_EXPS}")
    print(f"Saving results to: {OUTPUT_DIR}")
    print("-" * 60)

    results_E = []
    results_R = []
    
    t_start_total = time.time()

    for i, R in enumerate(BOND_LENGTHS):
        print(f"\n[Point {i+1}/{len(BOND_LENGTHS)}] R = {R:.4f} Bohr")
        
        # 1. Setup Molecule Geometry (Aligned along Z axis)
        # H at +z and -z. As R changes, nuclei move relative to the fixed DVR grid.
        z_pos = R / 2.0
        coords = np.array([
            [0.0, 0.0, -z_pos],
            [0.0, 0.0,  z_pos]
        ])
        charges = np.array([1.0, 1.0])
        mol = Molecule(charges, coords, nelec=NELEC)
        Enuc = mol.nuclear_repulsion_energy()
        
        # 2. Build Hamiltonian (Method 2)
        # This constructs integrals, DVR grid, and initial Guess (E_slices, C_list)
        try:
            Hcore, z_grid, dz, E_slices, C_list, ERI_J, ERI_K, shapes = build_method2(
                mol,
                Lz=LZ_VAL,
                Nz=NZ_VAL,
                M=M_VAL,
                s_exps=H_S_EXPS,
                p_exps=H_P_EXPS,
                d_exps=H_D_EXPS,
                max_offset=None, # Use default full range or auto-cut
                auto_cut=True,   # Enable auto-cut for speed if ERI drops off
                verbose=False,    # Reduce spam
                dvr_method='sine'
            )
            
            # 3. Run SCF (Pure RHF, no Newton)
            Etot, eps, Cmo, P, info = scf_rhf_method2(
                Hcore, ERI_J, ERI_K, NZ_VAL, M_VAL,
                nelec=NELEC, Enuc=Enuc,
                conv=1e-8, max_iter=100, verbose=False
            )
            
            print(f"  -> Converged: E = {Etot:.8f} Eh (iters={info['iter']})")
            
            results_R.append(R)
            results_E.append(Etot)

            # 4. Save detailed snapshot
            # Captures AO (C_list), MO (Cmo), and Scan Meta-data
            # We add 'bond_length' to the save by passing it manually to a modified saver 
            # or just creating a dict file here since save_scf_snapshot is fixed.
            # We will use the provided save_scf_snapshot and add a meta file or 
            # trust the filename tagging.
            
            # Re-creating primitive params for saving context
            # (In a real script, these are returned by make_xy..., but build_method2 hides them.
            #  However, save_scf_snapshot requires alphas/centers/labels. 
            #  We must regenerate them to pass to the saver, or skip saving them if not strictly needed.
            #  To be safe, let's regenerate the basis labels/alphas to pass to the saver function.)
            
            from pyqed.qchem.dvr.hybrid_gauss_dvr_integrals_add_d_orbitals import make_xy_spd_primitive_basis
            alphas, centers, labels = make_xy_spd_primitive_basis(
                mol.to_tuples(),
                exps_s=H_S_EXPS,
                exps_p=H_P_EXPS,
                exps_d=H_D_EXPS
            )

            save_label = f"R_{R:.4f}"
            save_scf_snapshot(
                run_folder=OUTPUT_DIR,
                run_label=save_label,
                Nz=NZ_VAL,
                M=M_VAL,
                cycle=0, # 0 implies pure SCF / no newton cycle
                Etot=Etot,
                C_list=C_list,
                Cmo=Cmo,
                P=P,
                eps=eps,
                info=info,
                alphas=alphas,
                centers=centers,
                labels=labels,
                z_grid=z_grid,
                Lz=LZ_VAL
            )
            
        except Exception as e:
            print(f"  -> Failed at R={R:.4f}: {e}")
            results_R.append(R)
            results_E.append(np.nan)

    # =============================================================================
    # ANALYSIS & PLOTTING
    # =============================================================================
    
    results_R = np.array(results_R)
    results_E = np.array(results_E)
    
    # Save summary text file
    summary_path = os.path.join(OUTPUT_DIR, "pes_summary.txt")
    np.savetxt(summary_path, np.column_stack((results_R, results_E)), header="R(Bohr)  Energy(Eh)")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(results_R, results_E, 'o-', markersize=4, label=f'Nz={NZ_VAL}, M={M_VAL}')
    plt.title(f"H2 Potential Energy Surface (Pure SCF)\nFixed M={M_VAL}, Nz={NZ_VAL}")
    plt.xlabel("Bond Length R (Bohr)")
    plt.ylabel("Total Energy (Hartree)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = os.path.join(OUTPUT_DIR, "pes_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nPlot saved to: {plot_path}")
    print(f"Total time: {time.time() - t_start_total:.2f} s")
    
    # Oscillation Check Hint
    # Calculate simple finite difference curvature to detect "noise"
    if len(results_E) > 4:
        dE = np.diff(results_E)
        ddE = np.diff(dE)
        noise_metric = np.std(ddE)
        print(f"Rough smoothness metric (std dev of 2nd deriv): {noise_metric:.2e}")
        print("(Lower is smoother. High values may indicate DVR grid oscillations.)")

if __name__ == "__main__":
    run_pes_scan()