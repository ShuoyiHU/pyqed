import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
from scipy.special import i0e

# ====================================================
#  1. CONSTANTS & DATA STRUCTURES
# ====================================================

@dataclass
class PrimitiveLabel:
    kind: str
    dim: int
    l: Tuple[int, int, int]
    role: str = "slice_2d"

# Prony Coefficients (Standard)
# ETAS = np.array([
#     0.04173077, 0.0972349, 0.12927493, 0.13390385, 0.1212556,
#     0.10187292, 0.0821291, 0.06475048, 0.05047605, 0.0391529,
#     0.03033575, 0.02354503, 0.01836517, 0.0144772, 0.01165111,
#     0.00969719, 0.00841568, 0.00761789, 0.00716047, 0.00695301
# ], dtype=np.float64)

ETAS = np.array([
    0.30069033, 0.16182767, 0.12407565, 0.09663122, 0.07452011,
    0.056974, 0.0433538, 0.03294897, 0.02506173, 0.0191048,
    0.01460944, 0.01121901, 0.00866935, 0.00677285, 0.00540124,
    0.00445656, 0.00384016, 0.00345875, 0.00324129, 0.00314306
], dtype=np.float64)

# XIS = np.array([
#     2.90966766e+00, 1.46957062e+00, 8.13288156e-01, 4.63366477e-01,
#     2.67827855e-01, 1.56348009e-01, 9.20302874e-02, 5.45820332e-02,
#     3.26022522e-02, 1.96041226e-02, 1.18613296e-02, 7.21519437e-03,
#     4.40506148e-03, 2.68870734e-03, 1.62584139e-03, 9.55255326e-04,
#     5.25214288e-04, 2.50901875e-04, 8.67029335e-05, 9.44699339e-06
# ], dtype=np.float64)

XIS = np.array([
    8.95803564e-01, 3.76331786e-01, 1.99974776e-01, 1.11372688e-01,
    6.32311898e-02, 3.63237834e-02, 2.10614754e-02, 1.23135953e-02,
    7.25491440e-03, 4.30560406e-03, 2.57258719e-03, 1.54631802e-03,
    9.33542237e-04, 5.64018239e-04, 3.38088583e-04, 1.97286602e-04,
    1.07942037e-04, 5.14001243e-05, 1.77285290e-05, 1.92999376e-06
], dtype=np.float64)
# ====================================================
#  2. HELPER FUNCTIONS
# ====================================================

def gamma_binom(n, k):
    if k < 0 or k > n: return 0
    if k == 0 or k == n: return 1
    if k > n // 2: k = n - k
    res = 1
    for i in range(k):
        res = res * (n - i) // (i + 1)
    return res

def _parse_2d_l(lbl: PrimitiveLabel) -> Tuple[int, int]:
    return lbl.l[0], lbl.l[1]

def make_xy_spd_primitive_basis(nuclei_tuples, exps_s, exps_p, exps_d):
    exps_s = np.asarray(exps_s, float).reshape(-1)
    exps_p = np.asarray(exps_p, float).reshape(-1)
    exps_d = np.asarray(exps_d, float).reshape(-1)
    
    rows = []
    for (Z, x, y, z) in nuclei_tuples:
        x, y = float(x), float(y)
        for a in exps_s: rows.append((a, x, y, 0))
        for a in exps_p: 
            rows.append((a, x, y, 1)) # px
            rows.append((a, x, y, 2)) # py
        for a in exps_d: 
            rows.append((a, x, y, 3)) # dx2
            rows.append((a, x, y, 4)) # dy2
            rows.append((a, x, y, 5)) # dxy

    if not rows: return np.array([]), np.array([]), []

    arr = np.array(rows, float)
    alphas  = arr[:, 0]
    centers = arr[:, 1:3]
    kinds   = arr[:, 3].astype(int)

    labels = []
    for k in kinds:
        if k == 0: labels.append(PrimitiveLabel("2d-s", 2, (0,0,0)))
        elif k == 1: labels.append(PrimitiveLabel("2d-px", 2, (1,0,0)))
        elif k == 2: labels.append(PrimitiveLabel("2d-py", 2, (0,1,0)))
        elif k == 3: labels.append(PrimitiveLabel("2d-dx2", 2, (2,0,0)))
        elif k == 4: labels.append(PrimitiveLabel("2d-dy2", 2, (0,2,0)))
        elif k == 5: labels.append(PrimitiveLabel("2d-dxy", 2, (1,1,0)))
        
    return alphas, centers, labels

# ====================================================
#  3. INTEGRAL ENGINES
# ====================================================

def _V_en_prony_general(alphas, centers, labels, nuc_xy, dz_abs):
    """ Standard Prony Expansion for V_en """
    invz = 1.0 / abs(dz_abs)
    n = len(alphas)
    out = np.zeros((n, n), dtype=float)
    xn, yn = nuc_xy

    for i in range(n):
        aA = alphas[i]; xA, yA = centers[i]; lxA, lyA = _parse_2d_l(labels[i])
        for j in range(i, n):
            aB = alphas[j]; xB, yB = centers[j]; lxB, lyB = _parse_2d_l(labels[j])
            
            gamma = aA + aB
            P_x = (aA * xA + aB * xB) / gamma
            P_y = (aA * yA + aB * yB) / gamma
            
            K_AB_x = np.exp(- aA*aB/gamma * (xA-xB)**2)
            K_AB_y = np.exp(- aA*aB/gamma * (yA-yB)**2)
            
            val_sum = 0.0
            for eta, xi in zip(ETAS, XIS):
                gam_p = xi * invz**2
                weight_p = eta * invz
                
                zeta = gamma + gam_p
                Qx = (gamma * P_x + gam_p * xn) / zeta
                Qy = (gamma * P_y + gam_p * yn) / zeta
                
                K_PN_x = np.exp(- gamma*gam_p/zeta * (P_x - xn)**2)
                K_PN_y = np.exp(- gamma*gam_p/zeta * (P_y - yn)**2)
                
                K_total = (K_AB_x * K_AB_y) * (K_PN_x * K_PN_y) * (np.pi / zeta) * weight_p
                
                def poly_int(l1, l2, q, c1, c2, zeta_val):
                    q1 = q - c1
                    q2 = q - c2
                    res = 0.0
                    for ii in range(l1 + 1):
                        term1 = gamma_binom(l1, ii) * (q1**(l1 - ii))
                        for jj in range(l2 + 1):
                            term2 = gamma_binom(l2, jj) * (q2**(l2 - jj))
                            k = ii + jj
                            if k % 2 == 0:
                                m = k // 2
                                dfact = 1.0
                                for x in range(1, 2*m, 2): dfact *= x
                                integral_part = dfact / ((2*zeta_val)**m)
                                res += term1 * term2 * integral_part
                    return res
                
                Ix = poly_int(lxA, lxB, Qx, xA, xB, zeta)
                Iy = poly_int(lyA, lyB, Qy, yA, yB, zeta)
                val_sum += K_total * Ix * Iy
                
            out[i, j] = val_sum
            out[j, i] = val_sum
    return out

def _V_en_exact_dz0_general_FIXED(alphas, centers, labels, nuc_xy):
    """ 
    Exact V_en at dz=0. 
    FIX: Fallback for p/d orbitals uses 1e-7 instead of 1e-2.
    """
    n = len(alphas)
    out = np.zeros((n, n), dtype=float)
    nx, ny = nuc_xy

    for i in range(n):
        ai = alphas[i]; xi, yi = centers[i]; lxi, lyi = _parse_2d_l(labels[i])
        for j in range(i, n):
            aj = alphas[j]; xj, yj = centers[j]; lxj, lyj = _parse_2d_l(labels[j])

            total_L = lxi + lyi + lxj + lyj
            gamma_val = ai + aj
            
            if total_L == 0:
                # Exact Analytical for S-S
                K_AB = np.exp( -ai*aj/gamma_val * ((xi-xj)**2 + (yi-yj)**2) )
                Px = (ai*xi + aj*xj)/gamma_val
                Py = (ai*yi + aj*yj)/gamma_val
                Dx = Px - nx; Dy = Py - ny
                D2 = Dx**2 + Dy**2
                x_arg = 0.5 * gamma_val * D2
                val = K_AB * (np.pi / np.sqrt(gamma_val)) * np.sqrt(np.pi) * i0e(x_arg)
                out[i,j] = val; out[j,i] = val
            else:
                # Fallback for P/D: Use tiny dz (1e-7)
                a_sub = np.array([ai, aj])
                c_sub = np.array([[xi, yi], [xj, yj]])
                l_sub = [labels[i], labels[j]]
                # This call handles the integral part
                mat = _V_en_prony_general(a_sub, c_sub, l_sub, nuc_xy, 1e-7)
                out[i,j] = mat[0,1]
                out[j,i] = mat[0,1]
    return out

def V_en_sp_total_extrapolated(alphas, centers, labels, nuclei_tuples, z):
    """
    The NEW Strategy: Constrained Parabolic Extrapolation
    """
    N = len(alphas)
    V = np.zeros((N, N), float)
    
    DZ_CUTOFF = 0.005
    
    for (Z_nuc, xA, yA, zA) in nuclei_tuples:
        dz = abs(z - zA)
        nuc_xy = np.array([xA, yA])

        if dz > DZ_CUTOFF:
            # Safe Region
            V_nuc = _V_en_prony_general(alphas, centers, labels, nuc_xy, dz)
        else:
            # Interpolation Region
            # 1. Anchor A: Exact Limit at z=0
            mat_exact = _V_en_exact_dz0_general_FIXED(alphas, centers, labels, nuc_xy)
            A_mat = mat_exact
            
            # 2. Anchor B: Prony at boundary 0.01
            mat_bound = _V_en_prony_general(alphas, centers, labels, nuc_xy, DZ_CUTOFF)
            
            # 3. Solve B = (V_bound - A) / cutoff^2
            B_mat = (mat_bound - A_mat) / (DZ_CUTOFF**2)
            
            # 4. Extrapolate
            V_nuc = A_mat + B_mat * (dz**2)
            
        V -= Z_nuc * V_nuc
        
    return V

# ====================================================
#  4. MAIN EXECUTION & PLOTTING
# ====================================================
if __name__ == "__main__":
    print("Running Extrapolation Test...")
    
    # 1. Setup Basis (S-orbitals only for clarity)
    s_exps = [18.73113696, 2.825394365, 0.6401216923, 0.1612777588]
    nuclei = [(1.0, 0.0, 0.0, 0.0)]
    alphas, centers, labels = make_xy_spd_primitive_basis(nuclei, s_exps, [], [])
    
    # We will track the first diagonal element
    idx = 0
    
    # 2. Define Grid (Logspace)
    dz_vals = np.logspace(-7, -1, 100)
    
    # 3. Compute Curves
    vals_prony_pure = []
    vals_new_method = []
    
    # Get Exact Limit for Reference
    mat_exact = _V_en_exact_dz0_general_FIXED(alphas, centers, labels, np.array([0.0, 0.0]))
    val_exact = -1.0 * mat_exact[idx, idx]
    
    for dz in dz_vals:
        # A. Pure Prony (Old Unstable Way) - forcing direct call
        mat_p = _V_en_prony_general(alphas, centers, labels, np.array([0.0, 0.0]), dz)
        vals_prony_pure.append(-1.0 * mat_p[idx, idx])
        
        # B. New Extrapolated Method
        # We pass z = dz (since atom is at 0)
        # Note: function returns total V, so we just take element
        mat_new = V_en_sp_total_extrapolated(alphas, centers, labels, nuclei, dz)
        vals_new_method.append(mat_new[idx, idx])

    # 4. Plot
    plt.figure(figsize=(10, 6))
    
    # Plot Pure Prony
    plt.plot(dz_vals, vals_prony_pure, 'b--', linewidth=1, label="Raw Prony (Unstable at small z)")
    
    # Plot New Method
    plt.plot(dz_vals, vals_new_method, 'r-', linewidth=2.5, alpha=0.8, label="Constrained Extrapolation (New)")
    
    # Plot Exact Limit
    plt.axhline(val_exact, color='g', linestyle=':', linewidth=2, label="Exact Analytical Limit (z=0)")
    
    # Plot Cutoff Line
    plt.axvline(0.01, color='k', linestyle='--', alpha=0.5, label="Cutoff (0.01)")
    
    # plt.xscale('log')
    plt.xlabel(r"$\Delta z$ (Bohr)")
    plt.ylabel("Potential Energy (Hartree)")
    plt.title("Comparison: Prony vs. Constrained Parabolic Extrapolation")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    # Save
    out_file = "extrapolation_test_result.png"
    plt.savefig(out_file, dpi=150)
    print(f"Test completed. Plot saved to {out_file}")
    plt.show()
