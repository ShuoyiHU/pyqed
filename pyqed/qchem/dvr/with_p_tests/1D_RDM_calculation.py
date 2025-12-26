import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import Tuple

# =================================================================================
#  PART 1: PRIMITIVE INTEGRALS & PHYSICS
# =================================================================================

@dataclass
class PrimitiveLabel:
    kind: str
    dim: int
    l: Tuple[int, int, int]

def gamma_binom(n, k):
    if k < 0 or k > n: return 0
    if k == 0 or k == n: return 1
    if k > n // 2: k = n - k
    res = 1
    for i in range(k):
        res = res * (n - i) // (i + 1)
    return res

def _overlap_1d_explicit(l1, l2, x1, x2, alpha, beta):
    gamma = alpha + beta
    P = (alpha * x1 + beta * x2) / gamma
    pre = np.exp(- (alpha * beta / gamma) * (x1 - x2)**2) * np.sqrt(np.pi / gamma)
    Q1, Q2 = P - x1, P - x2
    val = 0.0
    for i in range(l1 + 1):
        comb1 = gamma_binom(l1, i)
        for j in range(l2 + 1):
            comb2 = gamma_binom(l2, j)
            k = i + j
            if k % 2 == 0:
                m = k // 2
                dfact = 1.0
                for x in range(1, 2*m, 2): dfact *= x
                val += comb1 * comb2 * (Q1)**(l1-i) * (Q2)**(l2-j) * (dfact / ((2*gamma)**m))
    return pre * val

def calculate_primitive_overlap(alphas, centers, labels):
    N = len(alphas)
    S = np.zeros((N, N), float)
    parsed_labels = []
    for l in labels:
        if isinstance(l, dict): parsed_labels.append(PrimitiveLabel(l['kind'], l['dim'], tuple(l['l'])))
        elif hasattr(l, 'kind'): parsed_labels.append(l)
        else: parsed_labels.append(l)

    for i in range(N):
        for j in range(i, N):
            aA, aB = alphas[i], alphas[j]
            xA, yA = centers[i]
            xB, yB = centers[j]
            lxA, lyA = parsed_labels[i].l[0], parsed_labels[i].l[1]
            lxB, lyB = parsed_labels[j].l[0], parsed_labels[j].l[1]
            val = _overlap_1d_explicit(lxA, lxB, xA, xB, aA, aB) * \
                  _overlap_1d_explicit(lyA, lyB, yA, yB, aA, aB)
            S[i, j] = S[j, i] = val
    return S

def compute_adiabatic_1rdm(file_path):
    print(f"Loading {os.path.basename(file_path)}...")
    data = np.load(file_path, allow_pickle=True)
    P_full = data['P']         
    alphas = data['alphas']
    centers = data['centers']
    labels = data['labels_serialized']
    C_list = data['C_list']    
    Nz = int(data['Nz'])
    M = int(data['M'])
    Lz = float(data['Lz'])

    if C_list.dtype == object or isinstance(C_list, list):
        C_list = np.stack(C_list).astype(float)

    print(f"Calculating Primitive Overlap for {len(alphas)} primitives...")
    S_prim = calculate_primitive_overlap(alphas, centers, labels)
    P_tensor = P_full.reshape(Nz, M, Nz, M)
    
    print("Contracting Adiabatic Basis (Generating Gamma_kl)...")
    gamma_z = np.zeros((Nz, Nz))
    SC = np.zeros((Nz, len(alphas), M))
    for l in range(Nz):
        SC[l] = S_prim @ C_list[l]

    for k in range(Nz):
        for l in range(Nz):
            S_kl_eff = C_list[k].T @ SC[l]
            P_block = P_tensor[k, :, l, :]
            gamma_z[k, l] = np.sum(P_block * S_kl_eff.T) 

    return gamma_z, Lz, Nz

# =================================================================================
#  PART 2: REAL SPACE TRANSFORMATION (2D)
# =================================================================================

def sine_dvr_basis_val(z_eval, z_min, z_max, N, k_index):
    L = z_max - z_min
    j = k_index + 1
    pre_u = np.sqrt(2.0 / (N + 1))
    pre_f = np.sqrt(2.0 / L)
    n = np.arange(1, N + 1)
    arg = np.outer(n, np.pi * (z_eval - z_min) / L)
    f_n = pre_f * np.sin(arg)
    U_kn = pre_u * np.sin(np.pi * j * n / (N + 1))
    return U_kn @ f_n

def evaluate_spatial_1rdm_2d(gamma_matrix, Lz, Nz, z_points):
    """
    Transforms the discrete Gamma_{kl} matrix into the real-space Gamma(z, z')
    Gamma(z, z') = sum_{k,l} Gamma_{kl} * theta_k(z) * theta_l(z')
    
    Efficient Matrix Mult: Gamma_real = Theta.T @ Gamma_discrete @ Theta
    """
    print(f"Evaluating 1-RDM on {len(z_points)}x{len(z_points)} grid...")
    
    # 1. Build Theta Matrix: Shape (Nz, N_points)
    # Each column i is the vector of all basis functions evaluated at z_points[i]
    Theta = np.zeros((Nz, len(z_points)))
    for k in range(Nz):
        Theta[k, :] = sine_dvr_basis_val(z_points, -Lz, Lz, Nz, k)
        
    # 2. Matrix Multiplication
    # (N_pts, Nz) @ (Nz, Nz) @ (Nz, N_pts) -> (N_pts, N_pts)
    gamma_real = Theta.T @ gamma_matrix @ Theta
    
    return gamma_real

# =================================================================================
#  MAIN
# =================================================================================

def main():
    # -------------------------------------------------------------
    # EDIT PATH HERE
    # -------------------------------------------------------------
    file_path = "/Users/shuoyihu/Documents/Lab/Hybrid Gauss DVR/extrapolation_tests/data/H2/Nz_255_full/scan_results_task_1/R_0.7034/scf_res_R_0.7034_newton_Nz255_M1_cyc2_20251202_073401_705341.npz" 
    # -------------------------------------------------------------

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # 1. Compute Discrete Matrix
    gamma_discrete, Lz, Nz = compute_adiabatic_1rdm(file_path)
    
    # 2. Define High-Res Grid
    N_grid = 300
    z_evals = np.linspace(-Lz, Lz, N_grid)
    
    # 3. Transform to Real Space Surface
    gamma_real = evaluate_spatial_1rdm_2d(gamma_discrete, Lz, Nz, z_evals)
    
    # 4. Plot
    print("Generating 2D Heatmap...")
    plt.figure(figsize=(10, 8))
    
    # Use pcolormesh for correct axis mapping
    # Z, Z_prime meshgrid
    Z1, Z2 = np.meshgrid(z_evals, z_evals)
    
    # Plot absolute value (though 1-RDM should be real/symmetric for this system)
    # Using 'inferno' or 'viridis'. 'Seismic' is good if you expect negative phases.
    mesh = plt.pcolormesh(Z1, Z2, gamma_real, cmap='inferno', shading='auto')
    
    plt.colorbar(mesh, label="Correlation Magnitude $\gamma(z, z')$")
    plt.title(f"1-Electron Reduced Density Matrix (Real Space)\n{os.path.basename(file_path)}")
    plt.xlabel("Coordinate $z$ (bohr)")
    plt.ylabel("Coordinate $z'$ (bohr)")
    
    # Add diagonal line for reference (where density is defined)
    plt.plot([-Lz, Lz], [-Lz, Lz], color='white', linestyle='--', alpha=0.5, label='Diagonal $\\rho(z)$')
    plt.legend()
    
    outname = os.path.splitext(file_path)[0] + "_Full_1RDM_Surface.png"
    plt.savefig(outname, dpi=300)
    print(f"Saved plot to {outname}")
    plt.show()

if __name__ == "__main__":
    main()