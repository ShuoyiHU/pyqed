import numpy as np
import scipy.linalg as la
from pyqed.qchem.dvr.hybrid_gauss_dvr_integrals_add_p_orbitals import (
    make_xy_sp_primitive_basis,
    V_en_sp_total_at_z,
    overlap_2d_cartesian,
    kinetic_2d_cartesian,
    PrimitiveLabel,
    STO6_EXPS_H,
    Exp_631g_ss_H
)

# -----------------------------
#  Minimal Sine-DVR (Copied for standalone)
# -----------------------------
def sine_dvr_1d(zmin, zmax, N):
    L = zmax - zmin
    j = np.arange(1, N + 1)
    z = zmin + j * L / (N + 1)
    n = np.arange(1, N + 1)
    U = np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * np.outer(j, n) / (N + 1))
    lam = 0.5 * (np.pi * n / L) ** 2
    Tz = (U * lam) @ U.T
    return z, Tz

def build_hamiltonian_full(Nz, S_prim, T_prim, Tz, z_grid, alphas, centers, labels, nuclei):
    """
    Build the full (Nz*N_ao) x (Nz*N_ao) Hamiltonian matrix.
    H = Tz (kron) S_prim + I (kron) T_prim + V_en(z)
    """
    N_ao = len(alphas)
    size = Nz * N_ao
    
    # 1. Longitudinal Kinetic Energy: Tz (kron) S_prim
    H_long = np.kron(Tz, S_prim)
    
    # 2. Transverse Kinetic Energy: I (kron) T_prim
    H_trans = np.kron(np.eye(Nz), T_prim)
    
    # 3. Potential Energy: Block Diagonal V_en(z_n)
    H_pot = np.zeros((size, size))
    for n in range(Nz):
        # Get V_en matrix at slice z_n
        Vn = V_en_sp_total_at_z(alphas, centers, labels, nuclei, float(z_grid[n]))
        
        start = n * N_ao
        end = start + N_ao
        H_pot[start:end, start:end] = Vn
        
    # Total H
    H_total = H_long + H_trans + H_pot
    
    # Total Overlap Matrix S_total = I (kron) S_prim
    # (Since Sine-DVR basis is orthogonal in z)
    S_total = np.kron(np.eye(Nz), S_prim)
    
    return H_total, S_total

def solve_h_atom(s_exps, p_exps, label, shift_x=0.0):
    print(f"\n--- Solving Hydrogen Atom: {label} ---")
    print(f"    Nucleus Shift X = {shift_x:.2f} (Breaks Symmetry)")
    
    # 1. System Setup: Nucleus shifted if requested
    nuclei = [(1.0, shift_x, 0.0, 0.0)] 
    
    # 2. Basis Setup: Basis functions centered at ORIGIN
    # This creates an offset if shift_x != 0
    
    # We manually construct basis at (0,0) so we can keep it fixed while moving nucleus
    # This mimics the "atom in a molecule" environment where basis is not perfect
    
    # Actually, simpler: Put nucleus at origin, put basis centers at (0,0)
    # To break symmetry, we add a basis function slightly off center?
    # Or just shift nucleus.
    
    # Let's put nucleus at (shift_x, 0) and basis functions at (0,0)
    
    dummy_nuclei_for_basis = [(1.0, 0.0, 0.0, 0.0)]
    alphas, centers, labels = make_xy_sp_primitive_basis(
        dummy_nuclei_for_basis, 
        exps_s=np.asarray(s_exps, float), 
        exps_p=np.asarray(p_exps, float)
    )
    
    # 3. Transverse Integrals
    S_prim = overlap_2d_cartesian(alphas, centers, labels)
    T_prim = kinetic_2d_cartesian(alphas, centers, labels)
    
    # 4. Longitudinal Grid (DVR)
    Nz = 127
    Lz = 8.0
    z_grid, Tz = sine_dvr_1d(-Lz, Lz, Nz)
    
    # 5. Build Full Hamiltonian (V_en uses the actual shifted nucleus)
    H_total, S_total = build_hamiltonian_full(Nz, S_prim, T_prim, Tz, z_grid, alphas, centers, labels, nuclei)
    
    # 6. Diagonalize 
    w_S, _ = la.eigh(S_total)
    min_eig = np.min(w_S)
    if min_eig < 1e-12:
        mask = w_S > 1e-10
        X = _[..., mask] / np.sqrt(w_S[mask])
        H_proj = X.T @ H_total @ X
        eigvals = la.eigvalsh(H_proj)
    else:
        eigvals = la.eigvalsh(H_total, S_total)
        
    E0 = eigvals[0]
    print(f"  Ground State Energy: {E0:.6f} Eh")
    
    return E0

if __name__ == "__main__":
    s_exps = Exp_631g_ss_H 
    p_exps = [1.1] 
    
    # Case 1: Perfect Symmetry (Nucleus at Origin)
    print("\n[Test A] Centrosymmetric Atom")
    E_s_sym = solve_h_atom(s_exps, [], "S-Only", shift_x=0.0)
    E_sp_sym = solve_h_atom(s_exps, p_exps, "S + P", shift_x=0.0)
    
    # Case 2: Broken Symmetry (Nucleus Shifted 0.5 bohr)
    # This forces the electron to polarize to chase the nucleus
    print("\n[Test B] Perturbed Atom (Shift=0.5)")
    E_s_pert = solve_h_atom(s_exps, [], "S-Only", shift_x=0.5)
    E_sp_pert = solve_h_atom(s_exps, p_exps, "S + P", shift_x=0.5)
    
    print("\n=== Summary ===")
    print(f"Symm S-Only: {E_s_sym:.6f}")
    print(f"Symm S+P:    {E_sp_sym:.6f} (Should be same as S-Only)")
    print(f"Pert S-Only: {E_s_pert:.6f} (Higher due to misalignment)")
    print(f"Pert S+P:    {E_sp_pert:.6f}")
    
    if E_sp_pert < -0.6:
        print("\n[CONCLUSION] In the perturbed case, S+P energy collapsed below -0.5.")
        print("The p-orbitals allowed polarization, which triggered the z-collapse.")