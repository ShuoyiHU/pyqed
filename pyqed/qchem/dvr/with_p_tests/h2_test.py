import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyscf import gto, scf
import time
import os
import pyqed.qchem.dvr.hybrid_gauss_dvr_method_add_d_orbitals as dvr_method
from pyqed.qchem.dvr.hybrid_gauss_dvr_integrals_add_d_orbitals import (
    make_xy_spd_primitive_basis,
    eri_2d_cartesian_with_p,
    overlap_2d_cartesian,
    kinetic_2d_cartesian,
    V_en_sp_total_at_z,
    build_h1_nm
)
# Newton Helper (Assumed to be in pyqed path or accessible)
from pyqed.qchem.dvr.newton_helper import CollocatedERIOp, NewtonHelper

# -----------------------------------------------------------------------------
# 2. HELPER: PYSCF REFERENCE
# -----------------------------------------------------------------------------
def run_pyscf_h2(bond_len_ang, basis_name):
    """Calculates H2 energy using PySCF for reference lines."""
    # Convert half-bond length to Angstrom z-coord
    z = bond_len_ang / 2.0
    atom_str = f"H 0.0 0.0 {z}; H 0.0 0.0 {-z}"
    
    mol = gto.M(atom=atom_str, unit='Angstrom', basis=basis_name, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    return mf.e_tot

# -----------------------------------------------------------------------------
# 3. HELPER: DVR + NEWTON SOLVER
# -----------------------------------------------------------------------------
def solve_dvr_newton(bond_len_ang, use_p=False, verbose=False):
    """
    Runs the full Hybrid Gauss-DVR Method 2 with Newton Optimization.
    Replicates the logic from the __main__ block of the method file.
    """
    # -- Constants & Conversion --
    BOHR_TO_ANG = 0.529177210903
    r_bohr = bond_len_ang / BOHR_TO_ANG
    z_pos = r_bohr / 2.0
    
    # -- Molecule Setup --
    charges = [1.0, 1.0]
    coords = np.array([
        [0.0, 0.0,  z_pos],
        [0.0, 0.0, -z_pos]
    ])
    mol = dvr_method.Molecule(charges, coords)
    
    # -- Basis Set --
    # 6-31G s-exponents for H
    s_exps = [18.73113696, 2.825394365, 0.6401216923, 0.1612777588]
    d_exps = []
    
    if use_p:
        p_exps = [1.1] # As requested
    else:
        p_exps = []

    # -- DVR Parameters --
    Nz = 63
    Lz = 6.0
    M = 1
    
    # -- Newton Parameters --
    # Adapted from your method file
    ALT_CYCLES              = 3
    NEWTON_STEPS_PER_CYCLE  = 1
    ACTIVE_MODE             = "topk_grad"
    ACTIVE_TOPK             = Nz  # Optimize all slices
    NEWTON_RIDGE            = 0.0
    SHORT_SCF_MAXITER       = 60
    
    # 1. Build Method 2 (Initialization)
    # This sets up Hcore, ERI tensors, and initial guess
    Hcore, z_grid, dz, E_slices, C_list, ERI_J, ERI_K, shapes = dvr_method.build_method2(
        mol, Lz=Lz, Nz=Nz, M=M,
        s_exps=s_exps, p_exps=p_exps, d_exps=d_exps,
        verbose=False
    )
    
    # 2. Initial SCF
    Etot, eps, Cmo, P, info = dvr_method.scf_rhf_method2(
        Hcore, ERI_J, ERI_K, Nz, M,
        nelec=mol.nelec, Enuc=mol.nuclear_repulsion_energy(),
        conv=1e-7, max_iter=100, verbose=False
    )
    if verbose: print(f"    Initial SCF: {Etot:.6f}")

    # 3. Setup Newton Machinery
    nuclei = mol.to_tuples()
    alphas, centers, labels = make_xy_spd_primitive_basis(nuclei, s_exps, p_exps, d_exps)
    S_prim = overlap_2d_cartesian(alphas, centers, labels)
    T_prim = kinetic_2d_cartesian(alphas, centers, labels)
    
    # Generate K_h kernels for ERI operator
    n_ao = len(alphas)
    K_h = []
    Kx_h = []
    for h in range(Nz):
        dz_val = h * dz
        eri_tensor = eri_2d_cartesian_with_p(alphas, centers, labels, delta_z=dz_val)
        K_h.append(eri_tensor.reshape(n_ao*n_ao, n_ao*n_ao))
        eri_perm = eri_tensor.transpose(0, 2, 1, 3)
        Kx_h.append(eri_perm.reshape(n_ao*n_ao, n_ao*n_ao))
        
    # Build h1_nm operator (Kinetic + Nuclear)
    # Note: We need the z-grid basis (Kz) which is inside 'sine_dvr_1d'
    # but build_method2 returned 'z_grid' and 'Hcore', not Kz directly.
    # We must regenerate Kz locally to build the Newton operator.
    _, Kz, _ = dvr_method.sine_dvr_1d(-Lz, Lz, Nz)
    
    h1_nm = build_h1_nm(
        Kz, S_prim, T_prim, z_grid,
        lambda zz: V_en_sp_total_at_z(alphas, centers, labels, nuclei, zz)
    )

    ERIop = CollocatedERIOp.from_kernels(
        N=S_prim.shape[0], Nz=Nz, dz=dz, K_h=K_h, Kx_h=Kx_h
    )
    
    nh = NewtonHelper(h1_nm, S_prim, ERIop)
    
    # Prepare d_stack (coefficients) for optimization
    d_stack = np.vstack([C_list[n][:, 0].copy() for n in range(Nz)])
    for n in range(Nz): # Normalize
        dn = d_stack[n]
        d_stack[n] = dn / np.sqrt(float(dn.T @ (S_prim @ dn)))

    # 4. The Newton Master Loop
    best_E = Etot
    
    for cyc in range(ALT_CYCLES):
        P_slice = P.reshape(Nz, 1, Nz, 1)[:, 0, :, 0].copy()
        
        # Select Active Slices
        active = dvr_method.select_active_slices(nh, d_stack, P_slice, mode=ACTIVE_MODE, topk=ACTIVE_TOPK)
        
        # Newton Step
        g_full = nh.gradient(d_stack, P_slice)
        delta_dict, lam, info_kkt = nh.kkt_step(d_stack, P_slice, S_prim, active=active, ridge=NEWTON_RIDGE)
        
        # Ensure descent direction
        if dvr_method.g_dot_delta(g_full, delta_dict, active) > 0.0:
            for n in active: delta_dict[n] *= -1.0
            
        # Trust Region / Scaling
        delta_dict, scaled = dvr_method.bound_step_S(delta_dict, active, S_prim, 0.10)
        
        # Line Search
        STEP_LIST = [3.0,1.5, 1.0, 0.5, 0.2, 0.1, -1.0, -0.5, -0.2]
        step_best_local = (np.inf, None, None, None, None)
        
        for step in STEP_LIST:
            E_try, d_try, C_try, P_try = dvr_method.evaluate_trial_step(
                step, d_stack, C_list, active,
                S_prim, z_grid, Kz, T_prim, alphas, centers, labels, nuclei,
                Nz, M, mol.nelec, mol.nuclear_repulsion_energy(), K_h, Kx_h,
                SHORT_SCF_MAXITER, delta_dict
            )
            if E_try < step_best_local[0]:
                step_best_local = (E_try, step, d_try, C_try, P_try)
                
        # Update if better
        if step_best_local[0] < best_E - 1e-8:
            best_E  = step_best_local[0]
            d_stack = step_best_local[2]
            C_list  = step_best_local[3]
            P       = step_best_local[4]
            # Run a clean SCF at the new point
            Hcore_new = dvr_method.rebuild_Hcore_from_d(d_stack, z_grid, Kz, S_prim, T_prim, alphas, centers, labels, nuclei)
            ERI_J, ERI_K = dvr_method.eri_JK_from_kernels_M1(C_list, K_h, Kx_h)
            Etot, _, _, P, _ = dvr_method.scf_rhf_method2(
                 Hcore_new, ERI_J, ERI_K, Nz, M,
                 nelec=mol.nelec, Enuc=mol.nuclear_repulsion_energy(),
                 conv=1e-7, max_iter=100, verbose=False
            )
            best_E = Etot
        else:
            # Convergence reached or no good step
            break
            
    return best_E

# -----------------------------------------------------------------------------
# 4. MAIN SCAN LOOP
# -----------------------------------------------------------------------------
def main():
    print("=========================================================")
    print("   H2 PES SCAN: HYBRID-GAUSS-DVR (NEWTON) vs PYSCF")
    print("=========================================================")
    print(" Parameters: Nz=63, Lz=6.0, M=1")
    
    # Define Scan Range (Angstroms)
    r_vals = np.linspace(0.4, 2.5339, 64) # 15 points
    print(r_vals)
    data = {
        "R_Ang": [],
        "DVR_No_P": [],
        "DVR_With_P": [],
        "PySCF_631G": [],
        "PySCF_sto6g": []
    }
    
    t0 = time.time()
    
    for i, r in enumerate(r_vals):
        print(f"\n[{i+1}/{len(r_vals)}] Calculating R = {r:.3f} Angstrom")
        
        # 1. PySCF References
        e_py_s  = run_pyscf_h2(r, '6-31g')
        e_py_sp = run_pyscf_h2(r, 'sto6g')
        print(f"  PySCF 6-31G:   {e_py_s:.6f}")
        print(f"  PySCF sto6g: {e_py_sp:.6f}")
        
        # 2. DVR (s-only)
        e_dvr_s = solve_dvr_newton(r, use_p=False)
        print(f"  DVR (s-only):  {e_dvr_s:.6f}")
        
        # # 3. DVR (s+p)
        # e_dvr_sp = solve_dvr_newton(r, use_p=True)
        # print(f"  DVR (s+p):     {e_dvr_sp:.6f}")
        
        data["R_Ang"].append(r)
        data["PySCF_631G"].append(e_py_s)
        data["PySCF_sto6g"].append(e_py_sp)
        data["DVR_No_P"].append(e_dvr_s)
        # data["DVR_With_P"].append(e_dvr_sp)
        
    print(f"\nTotal Time: {time.time() - t0:.2f} seconds")
    
    # -------------------------------------------------------------------------
    # 5. PLOTTING
    # -------------------------------------------------------------------------
    df = pd.DataFrame(data)
    df.to_csv("h2_pes_scan_newton.csv", index=False)
    
    plt.figure(figsize=(10, 7))
     # Placeholder for the generated plot
    
    # PySCF References (Dashed lines)
    plt.plot(df["R_Ang"], df["PySCF_631G"], 'b--', label="PySCF 6-31G (Contracted)")
    plt.plot(df["R_Ang"], df["PySCF_sto6g"], 'r--', label="PySCF STO-6G (Contracted)")
    
    # DVR Results (Points + Line)
    plt.plot(df["R_Ang"], df["DVR_No_P"], 'o-', color='cyan', label="DVR (s-only, Uncontracted)")
    # plt.plot(df["R_Ang"], df["DVR_With_P"], 's-', color='orange', label="DVR (s + p=1.1, Uncontracted)")
    
    plt.title("H2 Potential Energy Surface: Hybrid-Gauss-DVR vs PySCF")
    plt.xlabel("Bond Length (Angstrom)")
    plt.ylabel("Total Energy (Hartree)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.18, -0.9) # Focus on the well
    
    plt.savefig("h2_pes_comparison.png", dpi=2000)
    print("Plot saved to h2_pes_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()