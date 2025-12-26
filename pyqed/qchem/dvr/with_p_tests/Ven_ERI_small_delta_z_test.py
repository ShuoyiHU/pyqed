import numpy as np
import sys
import os

# Adjust path to find the module if strictly importing as requested
# Assuming this script is running in the root or appropriate relative path
from pyqed.qchem.dvr import hybrid_gauss_dvr_integrals_add_d_orbitals as hgi

def get_fd_derivative(func, center_idx, axis_idx, centers, h=1e-4):
    """
    Computes central finite difference derivative of a function 
    with respect to a specific coordinate of a specific center.
    """
    orig_val = centers[center_idx, axis_idx]
    
    # f(x + h)
    centers[center_idx, axis_idx] = orig_val + h
    val_plus = func(centers)
    
    # f(x - h)
    centers[center_idx, axis_idx] = orig_val - h
    val_minus = func(centers)
    
    # Restore
    centers[center_idx, axis_idx] = orig_val
    
    return (val_plus - val_minus) / (2.0 * h)

def run_ven_test():
    print("\n" + "="*60)
    print("TEST: Electron-Nuclear Attraction (V_en) [s vs px]")
    print("Verifying relation: <px|V|s> == (1/2a) * d/dx <s|V|s>")
    print("="*60)
    print(f"{'dz':<12} | {'Analytic (P)':<15} | {'FD from (S)':<15} | {'Error':<12}")
    print("-" * 60)

    # Setup
    alpha = 1.5
    # Nucleus at origin (0,0,0) implied by passing (0,0) to function and varying z
    nuclei = [(1.0, 0.0, 0.0, 0.0)] # Z, x, y, z
    
    # We test at various Z planes (dz)
    z_values = [0.0, 1e-8, 1e-6, 1e-5, 1e-4,1e-3, 1e-2, 0.1, 1.0]

    for z_curr in z_values:
        # 1. Analytic Calculation <px | V | s>
        # Basis: Center A at (0,0), Center B at (1.0, 0.0)
        # We put px on A, s on B.
        alphas = np.array([alpha, alpha])
        centers = np.array([[0.0, 0.0], [1.0, 0.0]]) # A at origin, B shifted
        
        # Labels for Analytic: A=px, B=s
        lbl_p = hgi.PrimitiveLabel(kind='2d-px', dim=2, l=(1,0,0))
        lbl_s = hgi.PrimitiveLabel(kind='2d-s',  dim=2, l=(0,0,0))
        labels_ana = [lbl_p, lbl_s]
        
        V_ana_mat = hgi.V_en_sp_total_at_z(alphas, centers, labels_ana, nuclei, z_curr)
        val_ana = V_ana_mat[0, 1] # <px | V | s>

        # 2. Finite Difference Calculation d/dxA <s | V | s>
        # We define a wrapper to change center A's x-coord
        def compute_s_s(current_centers):
            labels_s = [lbl_s, lbl_s] # Both s
            V = hgi.V_en_sp_total_at_z(alphas, current_centers, labels_s, nuclei, z_curr)
            return V[0, 1]

        # Calculate derivative wrt Center A x-coord
        # P = 1/(2alpha) * d/dx(S)
        slope = get_fd_derivative(compute_s_s, 0, 0, centers.copy(), h=1e-5)
        val_fd = slope / (2 * alpha)

        err = abs(val_ana - val_fd)
        print(f"{z_curr:<12.1e} | {val_ana:<15.6f} | {val_fd:<15.6f} | {err:<12.2e}")

def run_eri_test():
    print("\n" + "="*60)
    print("TEST: Electron Repulsion (ERI) [(px s | s s)]")
    print("Verifying: (px s | s s) == (1/2a) * d/dxA (s s | s s)")
    print("="*60)
    print(f"{'Delta Z':<12} | {'Analytic (P)':<15} | {'FD from (S)':<15} | {'Error':<12}")
    print("-" * 60)

    # Setup
    alpha = 2.0
    # Quartet: A, B, C, D. 
    # We want (A_px B_s | C_s D_s).
    # To use FD, we shift A_x in (A_s B_s | C_s D_s).
    
    alphas = np.array([alpha, alpha, alpha, alpha])
    
    # Geometry: 
    # Bra pair (0,1) near origin. Ket pair (2,3) shifted.
    # Center 0 is the one we toggle p/s.
    centers_base = np.array([
        [0.0, 0.0],  # 0: The one we differentiate
        [0.5, 0.0],  # 1
        [0.0, 0.5],  # 2 (on other electron plane)
        [0.5, 0.5]   # 3 (on other electron plane)
    ])

    dz_values = [0.0, 1e-7, 1e-5, 1e-4,1e-3,4e-3,5e-3, 1e-2, 0.5, 2.0]

    # Labels
    lbl_s = hgi.PrimitiveLabel(kind='2d-s', dim=2, l=(0,0,0))
    lbl_px = hgi.PrimitiveLabel(kind='2d-px', dim=2, l=(1,0,0))

    for dz in dz_values:
        # 1. Analytic: (px s | s s)
        labels_ana = [lbl_px, lbl_s, lbl_s, lbl_s]
        eri_ana = hgi.eri_2d_cartesian_with_p(alphas, centers_base, labels_ana, dz)
        val_ana = eri_ana[0, 1, 2, 3]

        # 2. FD: (s s | s s) shifted
        def compute_ssss(curr_centers):
            labels_ref = [lbl_s, lbl_s, lbl_s, lbl_s]
            # dz is fixed for this iteration
            eri = hgi.eri_2d_cartesian_with_p(alphas, curr_centers, labels_ref, dz)
            return eri[0, 1, 2, 3]

        slope = get_fd_derivative(compute_ssss, 0, 0, centers_base.copy(), h=1e-5)
        val_fd = slope / (2 * alpha)

        err = abs(val_ana - val_fd)
        print(f"{dz:<12.1e} | {val_ana:<15.6f} | {val_fd:<15.6f} | {err:<12.2e}")

if __name__ == "__main__":
    try:
        run_ven_test()
        run_eri_test()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()