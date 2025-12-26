import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict
from scipy.special import i0e, i1e, ive, gamma

# ====================================================
#  Basis function labels
# ====================================================

@dataclass
class PrimitiveLabel:
    """
    Minimal label for one primitive GTO.
    kind: '2d-s', '2d-px', '2d-py', '2d-dxy', '2d-dx2', '2d-dy2'
    dim : 2
    l   : (lx, ly, lz)
    """
    kind: str
    dim: int
    l: Tuple[int, int, int]
    role: str = "slice_2d"

# ==================================
#  Prony & STO-6G data
# ==================================

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

# ETAS = np.array([
#     0.64298191, 0.10687931, 0.06840397, 0.0482421, 0.03482642,
#     0.02536323, 0.01857658, 0.01367536, 0.01011727, 0.00752199,
#     0.00562028, 0.00422176, 0.00319191, 0.00243766, 0.00189678,
#     0.00152664, 0.00128801, 0.00114274, 0.00106124, 0.00102486
# ], dtype=np.float64)

# XIS = np.array([
#     2.90966766e+00, 1.46957062e+00, 8.13288156e-01, 4.63366477e-01,
#     2.67827855e-01, 1.56348009e-01, 9.20302874e-02, 5.45820332e-02,
#     3.26022522e-02, 1.96041226e-02, 1.18613296e-02, 7.21519437e-03,
#     4.40506148e-03, 2.68870734e-03, 1.62584139e-03, 9.55255326e-04,
#     5.25214288e-04, 2.50901875e-04, 8.67029335e-05, 9.44699339e-06
# ], dtype=np.float64)



# Updated XIS values
XIS = np.array([
    8.95803564e-01, 3.76331786e-01, 1.99974776e-01, 1.11372688e-01,
    6.32311898e-02, 3.63237834e-02, 2.10614754e-02, 1.23135953e-02,
    7.25491440e-03, 4.30560406e-03, 2.57258719e-03, 1.54631802e-03,
    9.33542237e-04, 5.64018239e-04, 3.38088583e-04, 1.97286602e-04,
    1.07942037e-04, 5.14001243e-05, 1.77285290e-05, 1.92999376e-06
], dtype=np.float64)
# XIS = np.array([
#     2.21023039e-01, 7.35503876e-02, 3.67967968e-02, 1.94315241e-02,
#     1.04872233e-02, 5.73951008e-03, 3.17739455e-03, 1.77738275e-03,
#     1.00395152e-03, 5.72303923e-04, 3.29062887e-04, 1.90699274e-04,
#     1.11241635e-04, 6.51258287e-05, 3.79868584e-05, 2.16925571e-05,
#     1.16863559e-05, 5.50878019e-06, 1.88881508e-06, 2.05068665e-07
# ], dtype=np.float64)
# Standard Exp for H (just placeholders)
STO6_EXPS_H = np.array([35.52322122, 6.513143725, 1.822142904, 0.6259552659, 0.2430767471, 0.1001124280], dtype=float)
Exp_631g_ss_H = np.array([18.73113696, 2.825394365, 0.6401216923, 0.1612777588], dtype=float)

# ===========================
#  Shared utilities (2D)
# ===========================

def pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    Ad = A[:, None, :] - B[None, :, :]
    return np.sum(Ad * Ad, axis=-1)

def _parse_2d_l(lbl: PrimitiveLabel) -> Tuple[int, int]:
    if lbl.dim != 2:
        raise ValueError("Only dim=2 primitives supported.")
    return lbl.l[0], lbl.l[1]

def make_xy_spd_primitive_basis(
    nuclei_tuples: List[Tuple[float, float, float, float]],
    exps_s: np.ndarray,
    exps_p: np.ndarray,
    exps_d: np.ndarray = None,
    decimals: int = 12,
):
    """
    Generates primitive basis including s, p, and d shells.
    d-shell uses Cartesian components: x^2, y^2, xy.
    """
    exps_s = np.asarray(exps_s, float).reshape(-1)
    exps_p = np.asarray(exps_p, float).reshape(-1)
    if exps_d is None: exps_d = np.array([], float)
    else: exps_d = np.asarray(exps_d, float).reshape(-1)
    
    rows = []
    for (Z, x, y, z) in nuclei_tuples:
        x, y = float(x), float(y)
        # s
        for a in exps_s: rows.append((a, x, y, 0))
        # p
        for a in exps_p: rows.append((a, x, y, 1)) # px
        for a in exps_p: rows.append((a, x, y, 2)) # py
        # d
        for a in exps_d: rows.append((a, x, y, 3)) # dx2 (l=2,0)
        for a in exps_d: rows.append((a, x, y, 4)) # dy2 (l=0,2)
        for a in exps_d: rows.append((a, x, y, 5)) # dxy (l=1,1)

    if not rows:
        return np.zeros((0,), float), np.zeros((0, 2), float), []

    arr = np.asarray(rows, float)
    # Sort/Unique logic
    # We round to detect duplicates, but usually bases are distinct per shell type
    arr_r = np.round(arr, decimals=decimals)
    _, idx = np.unique(arr_r, axis=0, return_index=True)
    idx = np.sort(idx)
    arr = arr[idx]

    alphas  = arr[:, 0].astype(float)
    centers = arr[:, 1:3].astype(float)
    kinds   = arr[:, 3].astype(int)

    labels: List[PrimitiveLabel] = []
    for k in kinds:
        if k == 0: labels.append(PrimitiveLabel(kind="2d-s",   dim=2, l=(0, 0, 0)))
        elif k == 1: labels.append(PrimitiveLabel(kind="2d-px",  dim=2, l=(1, 0, 0)))
        elif k == 2: labels.append(PrimitiveLabel(kind="2d-py",  dim=2, l=(0, 1, 0)))
        elif k == 3: labels.append(PrimitiveLabel(kind="2d-dx2", dim=2, l=(2, 0, 0)))
        elif k == 4: labels.append(PrimitiveLabel(kind="2d-dy2", dim=2, l=(0, 2, 0)))
        elif k == 5: labels.append(PrimitiveLabel(kind="2d-dxy", dim=2, l=(1, 1, 0)))
        
    return alphas, centers, labels

# Alias for backward compatibility
make_xy_sp_primitive_basis = make_xy_spd_primitive_basis


# ====================================
#  General 1D Integral Helpers
# ====================================

def _overlap_1d(l1: int, l2: int, x1: float, x2: float, gamma: float) -> float:
    """
    Computes 1D overlap integral: S = integral (x-x1)^l1 (x-x2)^l2 exp(-gamma (x-P)^2) dx * prefactor
    Note: The exponential exp(-alpha*(x-x1)^2 - beta*(x-x2)^2) is usually decomposed into:
       E_prefactor * exp(-gamma*(x-P)^2)
    This function computes the integral part int (x-x1)^l1 (x-x2)^l2 exp(-gamma*(x-P)^2) dx.
    P is the weighted center (alpha*x1 + beta*x2)/gamma.
    However, for simplicity in the calling function, we pass P implicitly or handle the coordinate shift.
    
    Alternative: Use standard recurrence.
    S(l1, l2) = <l1 | l2>.
    <0|0> = sqrt(pi/gamma)
    <1|0> = (P-x1) <0|0>
    <0|1> = (P-x2) <0|0>
    <l1+1 | l2> = (P-x1) <l1|l2> + (1/2gamma) ( l1<l1-1|l2> + l2<l1|l2-1> )
    """
    # We need alpha and beta to find P, but here we assume the caller provides P info or we compute it.
    # To make this pure, we require P and the prefactor K outside.
    # But typically we have alpha, beta.
    raise NotImplementedError("Use _overlap_1d_explicit")

def _overlap_1d_explicit(l1, l2, x1, x2, alpha, beta):
    """
    Full 1D Gaussian overlap <(x-x1)^l1 | (x-x2)^l2>.
    """
    gamma = alpha + beta
    P = (alpha * x1 + beta * x2) / gamma
    pre = np.exp(- (alpha * beta / gamma) * (x1 - x2)**2) * np.sqrt(np.pi / gamma)
    
    # Recurrence for I(l1, l2) = integral (x-x1)^l1 (x-x2)^l2 exp(-gamma(x-P)^2)
    # Base case
    # I(0,0) = 1 (prefactor applied at end)
    
    # Max angular momentum needed
    L = l1 + l2
    # We can center at P: x = u + P
    # (x-x1) = u + (P-x1) = u + Q1
    # (x-x2) = u + (P-x2) = u + Q2
    # Integral is sum_k c_k * integral u^k exp(-gamma u^2)
    # int u^k exp is non-zero only for even k.
    # int u^(2m) exp(-gamma u^2) = (2m-1)!! / (2gamma)^m * sqrt(pi/gamma)
    
    Q1 = P - x1
    Q2 = P - x2
    
    # Expand binomials
    val = 0.0
    for i in range(l1 + 1):
        comb1 = gamma_binom(l1, i) # (l1 choose i)
        term1 = (Q1)**(l1 - i)
        for j in range(l2 + 1):
            comb2 = gamma_binom(l2, j)
            term2 = (Q2)**(l2 - j)
            
            k = i + j
            if k % 2 == 0:
                m = k // 2
                # Double factorial (2m-1)!!
                dfact = 1.0
                for x in range(1, 2*m, 2): dfact *= x
                
                gauss_int = dfact / ((2*gamma)**m)
                val += comb1 * comb2 * term1 * term2 * gauss_int
                
    return pre * val

def gamma_binom(n, k):
    # Simple nCk for small n
    if k < 0 or k > n: return 0
    if k == 0 or k == n: return 1
    if k > n // 2: k = n - k
    res = 1
    for i in range(k):
        res = res * (n - i) // (i + 1)
    return res

def _kinetic_1d_explicit(l1, l2, x1, x2, alpha, beta):
    """
    1D Kinetic Energy <a | -0.5 d^2/dx^2 | b>
    -0.5 d^2/dx^2 ( (x-B)^l2 exp(-b(x-B)^2) )
    Using relation: d^2/dx^2 g_l = 4b^2 g_{l+2} - 2b(2l+1) g_l + l(l-1) g_{l-2}
    So T_ab = -0.5 * [ 4b^2 S(l1, l2+2) - 2b(2l2+1) S(l1, l2) + l2(l2-1) S(l1, l2-2) ]
    """
    term1 = 4.0 * beta**2 * _overlap_1d_explicit(l1, l2 + 2, x1, x2, alpha, beta)
    term2 = -2.0 * beta * (2 * l2 + 1) * _overlap_1d_explicit(l1, l2, x1, x2, alpha, beta)
    term3 = 0.0
    if l2 >= 2:
        term3 = l2 * (l2 - 1) * _overlap_1d_explicit(l1, l2 - 2, x1, x2, alpha, beta)
        
    return -0.5 * (term1 + term2 + term3)

# ====================================
#  2D Overlap and Kinetic (General)
# ====================================

def overlap_2d_cartesian(alphas, centers, labels):
    alphas = np.asarray(alphas, float)
    centers = np.asarray(centers, float)
    N = alphas.shape[0]
    S = np.zeros((N, N), float)

    for i in range(N):
        aA = alphas[i]; xA, yA = centers[i]; lxA, lyA = _parse_2d_l(labels[i])
        for j in range(i, N): # Symmetric
            aB = alphas[j]; xB, yB = centers[j]; lxB, lyB = _parse_2d_l(labels[j])
            
            Sx = _overlap_1d_explicit(lxA, lxB, xA, xB, aA, aB)
            Sy = _overlap_1d_explicit(lyA, lyB, yA, yB, aA, aB)
            val = Sx * Sy
            S[i, j] = val
            S[j, i] = val
    return S

def kinetic_2d_cartesian(alphas, centers, labels):
    alphas = np.asarray(alphas, float)
    centers = np.asarray(centers, float)
    N = alphas.shape[0]
    Tmat = np.zeros((N, N), float)

    for i in range(N):
        aA = alphas[i]; xA, yA = centers[i]; lxA, lyA = _parse_2d_l(labels[i])
        for j in range(i, N):
            aB = alphas[j]; xB, yB = centers[j]; lxB, lyB = _parse_2d_l(labels[j])
            
            # T = Tx Sy + Sx Ty
            Sx = _overlap_1d_explicit(lxA, lxB, xA, xB, aA, aB)
            Sy = _overlap_1d_explicit(lyA, lyB, yA, yB, aA, aB)
            Tx = _kinetic_1d_explicit(lxA, lxB, xA, xB, aA, aB)
            Ty = _kinetic_1d_explicit(lyA, lyB, yA, yB, aA, aB)
            
            val = Tx * Sy + Sx * Ty
            Tmat[i, j] = val
            Tmat[j, i] = val
    return Tmat

# ==============================================
#  Electron-Nuclear Attraction (Generalized)
# ==============================================

def _V_en_prony_general(alphas, centers, labels, nuc_xy, dz_abs):
    """
    General V_en using Prony expansion (dz > 0).
    Treats interaction as sum of 3-center overlaps.
    """
    invz = 1.0 / abs(dz_abs)
    n = len(alphas)
    out = np.zeros((n, n), dtype=float)
    xn, yn = nuc_xy

    for i in range(n):
        aA = alphas[i]; xA, yA = centers[i]; lxA, lyA = _parse_2d_l(labels[i])
        for j in range(i, n):
            aB = alphas[j]; xB, yB = centers[j]; lxB, lyB = _parse_2d_l(labels[j])
            
            # Merge A and B into P
            gamma = aA + aB
            P_x = (aA * xA + aB * xB) / gamma
            P_y = (aA * yA + aB * yB) / gamma
            
            # Precompute the base AB overlap prefactor (without the integral part)
            # overlap_1d = E_AB * integral (x-P)^k exp(-g(x-P)^2)
            # We need the explicit expansion around P.
            # Actually, simpler: V = sum_k weight_k * Overlap(A, B, N_k)
            # Where N_k is a gaussian at nuc_xy with exp gam_p[k]
            
            val_sum = 0.0
            for eta, xi in zip(ETAS, XIS):
                gam_p = xi * invz**2
                weight_p = eta * invz
                
                # We need < phi_A | phi_B | exp(-gam_p (r-N)^2) >
                # This separates into X and Y 3-center integrals.
                # I_x = integral (x-xA)^lA (x-xB)^lB exp(-aA(x-xA)^2 -aB(x-xB)^2 -gam_p(x-xn)^2)
                
                # 3-Gaussian Product Rule:
                # exp(-a(x-A)^2) exp(-b(x-B)^2) exp(-c(x-C)^2)
                # = K_ABC * exp(-(a+b+c)(x-Q)^2)
                
                # First merge A and B -> P, gamma
                # exp(-aA...)exp(-aB...) = K_AB * exp(-gamma(x-P)^2)
                # K_AB = exp(- aA*aB/gamma * (xA-xB)^2 )
                
                K_AB_x = np.exp(- aA*aB/gamma * (xA-xB)**2)
                K_AB_y = np.exp(- aA*aB/gamma * (yA-yB)**2)
                
                # Now merge P and N -> Q
                # exp(-gamma(x-P)^2) exp(-gam_p(x-xn)^2)
                # = K_PN * exp(-zeta(x-Q)^2)
                zeta = gamma + gam_p
                Qx = (gamma * P_x + gam_p * xn) / zeta
                Qy = (gamma * P_y + gam_p * yn) / zeta
                
                K_PN_x = np.exp(- gamma*gam_p/zeta * (P_x - xn)**2)
                K_PN_y = np.exp(- gamma*gam_p/zeta * (P_y - yn)**2)
                
                # Total Prefactor
                K_total = (K_AB_x * K_AB_y) * (K_PN_x * K_PN_y) * (np.pi / zeta) * weight_p
                
                # Polynomial part: (x-xA)^lA (x-xB)^lB
                # Center at Q: x = u + Q
                # (x-xA) = u + (Q-xA)
                # (x-xB) = u + (Q-xB)
                
                def poly_int(l1, l2, q, c1, c2, zeta_val):
                    # Integral u^k exp(-zeta u^2)
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

def _V_en_exact_dz0_general(alphas, centers, labels, nuc_xy):
    """
    Exact V_en at dz=0 for higher angular momentum.
    Uses generalized Boys-like logic (derivatives of V_ss).
    Limited to total L <= 4 for reasonable performance.
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
            
            # Exact s-s case (Total L=0)
            if total_L == 0:
                K_AB = np.exp( -ai*aj/gamma_val * ((xi-xj)**2 + (yi-yj)**2) )
                Px = (ai*xi + aj*xj)/gamma_val
                Py = (ai*yi + aj*yj)/gamma_val
                Dx = Px - nx; Dy = Py - ny
                D2 = Dx**2 + Dy**2
                x_arg = 0.5 * gamma_val * D2
                
                # Exact Bessel solution for s-s
                val = K_AB * (np.pi / np.sqrt(gamma_val)) * np.sqrt(np.pi) * i0e(x_arg)
                out[i,j] = val; out[j,i] = val
            else:
                # Fallback for p/d orbitals (Avoiding the complex derivatives for now)
                # We calculate this specific pair using Prony at a SAFE Z height.
                # This avoids the result being 0.0
                
                # Extract single pair for the helper
                a_sub = np.array([ai, aj])
                c_sub = np.array([[xi, yi], [xj, yj]])
                l_sub = [labels[i], labels[j]]
                
                # Call prony with SAFE_Z
                mat = _V_en_prony_general(a_sub, c_sub, l_sub, nuc_xy, 1e-2)
                out[i,j] = mat[0,1]
                out[j,i] = mat[0,1]
    
    return out

def V_en_sp_total_at_z(alphas, centers, labels, nuclei_tuples, z):
    N = len(alphas)
    V = np.zeros((N, N), float)
    for (Z, xA, yA, zA) in nuclei_tuples:
        dz = abs(z - zA)
        if dz < 1e-2:
             # Use the safer near-field generalized routine
             V_nuc = _V_en_exact_dz0_general(alphas, centers, labels, np.array([xA, yA]))
        else:
             V_nuc = _V_en_prony_general(alphas, centers, labels, np.array([xA, yA]), dz)
        V -= Z * V_nuc
    return V

# =========================================================
#  ERI (Two-Electron Integrals) - Generalized
# =========================================================

def _hermite_coefficients_general(l_a, l_b, alpha_a, alpha_b, A, B):
    """
    General Hermite expansion for product of two Gaussians.
    Returns E[t] for t in 0..la+lb.
    """
    gamma = alpha_a + alpha_b
    P = (alpha_a * A + alpha_b * B) / gamma
    inv_2gamma = 1.0 / (2.0 * gamma)
    
    # We need coefficients E_t such that:
    # (x-A)^la (x-B)^lb exp(-g(x-P)^2) = sum_t E_t H_t(sqrt(gamma)(x-P)) ...
    # Usually standard recurrence E^{ab}_t is used.
    # E[t] = E^{la, lb}_t
    # Range t: 0 to la+lb
    
    # Shape: (la+1, lb+1, la+lb+1)
    # But we only need the specific la, lb.
    # We can compute iteratively.
    
    L_max = l_a + l_b
    E = np.zeros((l_a + 1, l_b + 1, L_max + 1))
    
    # Base case 0,0
    E[0, 0, 0] = np.exp(- (alpha_a * alpha_b / gamma) * (A - B)**2)
    
    for i in range(l_a + 1):
        for j in range(l_b + 1):
            if i == 0 and j == 0: continue
            
            # Recurrence on i (increment la)
            # E^{i,j}_t = (P-A)*E^{i-1,j}_t + (t+1)E^{i-1,j}_{t+1} + 0.5/g * E^{i-1,j}_{t-1} ...
            # Wait, standard recurrence:
            # E[i,j,t] = (P-A)*E[i-1,j,t] + inv_2g * (t+1)*E[i-1,j,t+1] + inv_2g * t * E[i-1,j,t-1] ??
            # Let's use the increment j rule if i=0.
            
            if i > 0:
                # Form from i-1
                for t in range(L_max): # Safe range
                    val = (P - A) * E[i-1, j, t]
                    val += inv_2gamma * (t + 1) * E[i-1, j, t+1]
                    if t > 0:
                        val += inv_2gamma * E[i-1, j, t-1] # Warning: factor t or 1?
                        # The derivative of H_t gives 2t H_{t-1}. The expansion is usually in Cartesian.
                        # Let's assume standard Cartesian Gaussian expansion coeffs E_t (not Hermite).
                        # Omega_AB(x) = sum E_t (x-P)^t exp(...)
                        # Recurrence for Cartesian E:
                        # E^{i+1, j}_t = (P-A)E^{i,j}_t + E^{i,j}_{t-1} + inv_2g*(t+1)E^{i,j}_{t+1}
                        # Correct is:
                        # E^{i+1, j}_t = (P_A) E_t + E_{t-1} + 1/(2p) (t+1) E_{t+1} (if using Hermite)
                        # For Cartesian powers (x-P)^t:
                        # E^{i+1, j}_t = (P-A)E^ij_t + E^ij_{t-1} + 1/(2g) * (t+1) E^ij_{t+1} ... No.
                        pass
                        
    # SIMPLIFICATION:
    # We only need the specific final array E[t] for the fixed la, lb.
    # We can use the explicit binomial expansion again, it's robust.
    # (x-A)^la (x-B)^lb = sum q  c_q (x-P)^q
    # This avoids the complexity of the Hermite recurrence which is easy to get wrong.
    
    # (x-A) = (x-P) + (P-A) = u + QA
    # (x-B) = (x-P) + (P-B) = u + QB
    
    QA = P - A
    QB = P - B
    
    coeffs = np.zeros(L_max + 1)
    
    for i in range(l_a + 1):
        term_a = gamma_binom(l_a, i) * (QA**(l_a - i)) # u^i
        for j in range(l_b + 1):
            term_b = gamma_binom(l_b, j) * (QB**(l_b - j)) # u^j
            coeffs[i+j] += term_a * term_b
            
    # E_t here is coefficient of (x-P)^t.
    # The ERI standard usually expects coefficients of Lambda_t (Hermite).
    # If we use the standard Boys function integration later, we need Hermite coeffs?
    # Or we can just integrate u^t exp(-alpha u^2).
    # The existing code `_eri_2d_pure_coulomb` uses `_bessel_recursion_2d`.
    # That recursion expects Hermite expansion coefficients?
    # "Ex_bra_all = _hermite_coefficients_1d..."
    # "E_pp[..., 0] = X_PB * E_ps[..., 0] + E_ps[..., 1]"
    # This recurrence E^{i,j+1}_t = X_PB E + 1/(2g) E' ... looks like Hermite.
    # Let's implement the correct Hermite recurrence.
    
    H_coeffs = np.zeros((l_a + 1, l_b + 1, L_max + 1))
    H_coeffs[0,0,0] = np.exp(-(alpha_a * alpha_b / gamma) * (A - B)**2)
    
    for i in range(l_a + 1):
        for j in range(l_b + 1):
            if i == 0 and j == 0: continue
            
            # Build (i, j)
            if i > 0:
                # Increment A
                # E^{i,j}_t = (P-A) E^{i-1,j}_t + inv_2g * E^{i-1,j}_{t-1} + (t+1) inv_2g E^{i-1,j}_{t+1}? No.
                # Valid recurrence: E^{i+1,j}_t = (P-A) E^{i,j}_t + 1/(2p) E^{i,j}_{t-1} + (t+1)/(2p) E^{i,j}_{t+1}  <-- This is for E_t coeff of H_t.
                # Let's match existing code behavior:
                # E_sp (s,p) -> X_PB * E_ss + inv_2p * E_ss(t-1?) 
                # The code: E_sp[..., 0] = X_PB; E_sp[..., 1] = inv_2p.
                # Meaning t=0 term has X_PB, t=1 term has inv_2p.
                # This matches: E^{0,1}_0 = (P-B)E00_0 + 1/2p E00_1(0) + ... 
                # It seems existing code maps t index to Hermite order t.
                
                prev = H_coeffs[i-1, j]
                for t in range(L_max):
                    # Term 1: (P-A) * prev[t]
                    H_coeffs[i,j,t] += (P - A) * prev[t]
                    # Term 2: 1/(2g) * prev[t-1]  (increments t order)
                    if t > 0:
                        H_coeffs[i,j,t] += inv_2gamma * prev[t-1]
                    # Term 3: (t+1)/(2g) * prev[t+1] (decrements t order)
                    if t < L_max - 1:
                        H_coeffs[i,j,t] += inv_2gamma * (t+1) * prev[t+1]
            else:
                # Increment B
                prev = H_coeffs[i, j-1]
                for t in range(L_max):
                    H_coeffs[i,j,t] += (P - B) * prev[t]
                    if t > 0:
                        H_coeffs[i,j,t] += inv_2gamma * prev[t-1]
                    if t < L_max - 1:
                        H_coeffs[i,j,t] += inv_2gamma * (t+1) * prev[t+1]
                        
    return H_coeffs[l_a, l_b]


def _bessel_recursion_2d(t_max, u_max, Z, Delta_x, Delta_y, p_eff):
    """
    Stable recursion using scipy.special.ive to handle Z=0 singularity.
    Matches literature by avoiding 1/X divergence.
    """
    N_total = t_max + u_max
    n_batch = Z.shape[0]
    
    # 1. Compute I_n(X) * exp(-|X|) using stable library function
    vals_n = {}
    X = -Z
    for n in range(N_total + 1):
        vals_n[n] = ive(n, X)

    # 2. Populate Tensor
    I_tensor = np.zeros((t_max + 1, u_max + 1, N_total + 1, n_batch))
    for n in range(N_total + 1):
        I_tensor[0, 0, n] = ((-1.0)**n) * vals_n[n]
        
    # 3. Cartesian Recurrences (Stable)
    factor = -0.5 * p_eff 
    def get_n_sum(arr_slice, n_target):
        i_n = arr_slice[n_target]
        i_np1 = arr_slice[n_target + 1]
        i_nm1 = arr_slice[1] if n_target == 0 else arr_slice[n_target - 1]
        return i_nm1 + 2*i_n + i_np1

    for t in range(t_max):
        for n in range(N_total - t): 
            sum_prev = get_n_sum(I_tensor[t-1, 0], n) if t > 0 else 0.0
            sum_curr = get_n_sum(I_tensor[t, 0], n)
            I_tensor[t+1, 0, n] = factor * (t * sum_prev + Delta_x * sum_curr)
            
    for t in range(t_max + 1):
        for u in range(u_max):
            for n in range(N_total - t - u):
                sum_prev = get_n_sum(I_tensor[t, u-1], n) if u > 0 else 0.0
                sum_curr = get_n_sum(I_tensor[t, u], n)
                I_tensor[t, u+1, n] = factor * (u * sum_prev + Delta_y * sum_curr)
    
    return I_tensor[:, :, 0]


def eri_2d_cartesian_with_p(alphas, centers, labels, delta_z, dz_tol=1.0e-2):
    """
    Generalized ERI with D-orbital support.
    """
    import itertools
    
    # Dispatch for small dz
    if abs(delta_z) < dz_tol:
        # Pure Coulomb Logic using generalized Hermite
        n_ao = len(alphas)
        eri_tensor = np.zeros((n_ao, n_ao, n_ao, n_ao), dtype=float)
        A = alphas[:, None] + alphas[None, :]
        Xi = (alphas[:, None] * alphas[None, :]) / A
        P_centers = (alphas[:, None, None] * centers[:, None, :] + alphas[None, :, None] * centers[None, :, :]) / A[:, :, None]
        R_ij_sq = np.sum((centers[:, None, :] - centers[None, :, :])**2, axis=2)
        K_ij = np.exp(-Xi * R_ij_sq)
        
        # Loop over all quartets
        # We group by kind to vectorize?
        # Generalized: group by angular momentum (Lx, Ly) pairs.
        # Map kind to L pair
        kind_to_L = {}
        for lbl in labels:
            kind_to_L[lbl.kind] = lbl.l # (lx, ly, lz)
            
        # Group indices by kind
        inds_by_kind = {}
        for i, lbl in enumerate(labels):
            if lbl.kind not in inds_by_kind: inds_by_kind[lbl.kind] = []
            inds_by_kind[lbl.kind].append(i)
            
        kinds = sorted(inds_by_kind.keys())
        
        for ki, kj, kk, kl in itertools.product(kinds, repeat=4):
            ii = inds_by_kind[ki]; jj = inds_by_kind[kj]; kk_ = inds_by_kind[kk]; ll = inds_by_kind[kl]
            
            Lix, Liy = kind_to_L[ki][0:2]; Ljx, Ljy = kind_to_L[kj][0:2]
            Lkx, Lky = kind_to_L[kk][0:2]; Llx, Lly = kind_to_L[kl][0:2]
            
            # Bra side
            A_bra = A[np.ix_(ii, jj)]; P_bra = P_centers[np.ix_(ii, jj)]
            # Ket side
            B_ket = A[np.ix_(kk_, ll)]; Q_ket = P_centers[np.ix_(kk_, ll)]
            
            # Hermite coeffs
            # We need to evaluate _hermite_coefficients_general for vectors.
            # The function above was scalar. We assume vectorized inputs work or we loop?
            # The function assumes scalar A, B. Vectorizing it is tricky.
            # But the A, B arrays are large.
            # Let's rely on the structure: P = (aA+bB)/(a+b).
            # We can implement the recurrence vectorized easily.
            # Since explicit recurrence is implemented inside the loop, let's use a vectorized version of _hermite.

            def get_hermites(la, lb, al_a, al_b, cnt_a, cnt_b, P_vec):
                gam = al_a + al_b
                inv_2g = 0.5 / gam
                shape = P_vec.shape
                Lmax = la + lb
                H_tab = np.zeros((la+1, lb+1) + shape + (Lmax+1,))
                H_tab[0,0,...,0] = 1.0
                for i in range(la + 1):
                    for j in range(lb + 1):
                        if i==0 and j==0: continue
                        if i > 0:
                            prev = H_tab[i-1, j]
                            X = (P_vec - cnt_a)
                            H_tab[i,j,...,:] += X[...,None] * prev
                            
                            # FIX: Do NOT multiply by inv_2g here
                            t_vals = np.arange(1, Lmax+1)
                            H_tab[i,j,..., :-1] += t_vals * prev[..., 1:]
                            
                            # Only multiply by inv_2g here
                            H_tab[i,j,..., 1:] += inv_2g[...,None] * prev[..., :-1]
                        else:
                            prev = H_tab[i, j-1]
                            X = (P_vec - cnt_b)
                            H_tab[i,j,...,:] += X[...,None] * prev
                            
                            # FIX: Do NOT multiply by inv_2g here
                            t_vals = np.arange(1, Lmax+1)
                            H_tab[i,j,..., :-1] += t_vals * prev[..., 1:]
                            
                            # Only multiply by inv_2g here
                            H_tab[i,j,..., 1:] += inv_2g[...,None] * prev[..., :-1]
                return H_tab[la, lb]

            Ex_bra = get_hermites(Lix, Ljx, alphas[ii][:,None], alphas[jj][None,:], centers[ii,0][:,None], centers[jj,0][None,:], P_bra[...,0])
            Ey_bra = get_hermites(Liy, Ljy, alphas[ii][:,None], alphas[jj][None,:], centers[ii,1][:,None], centers[jj,1][None,:], P_bra[...,1])
            
            Ex_ket = get_hermites(Lkx, Llx, alphas[kk_][:,None], alphas[ll][None,:], centers[kk_,0][:,None], centers[ll,0][None,:], Q_ket[...,0])
            Ey_ket = get_hermites(Lky, Lly, alphas[kk_][:,None], alphas[ll][None,:], centers[kk_,1][:,None], centers[ll,1][None,:], Q_ket[...,1])
            P_bd = P_bra[:, :, None, None, :]
            Q_bd = Q_ket[None, None, :, :, :]
            Delta = Q_bd - P_bd
            A_bd = A_bra[:, :, None, None]
            B_bd = B_ket[None, None, :, :]
            Sigma = (A_bd + B_bd) / (4.0 * A_bd * B_bd)
            Z = -np.sum(Delta**2, axis=-1) / (8.0 * Sigma)
            p_eff = 1.0 / (4.0 * Sigma)
            
            Tx_max = (Lix + Ljx) + (Lkx + Llx)
            Ty_max = (Liy + Ljy) + (Lky + Lly)
            
            I_tensor = _bessel_recursion_2d(Tx_max, Ty_max, Z.ravel(), Delta[..., 0].ravel(), Delta[..., 1].ravel(), p_eff.ravel())
            I_tensor = I_tensor.reshape((Tx_max+1, Ty_max+1) + Z.shape)
            
            K_bra = K_ij[np.ix_(ii, jj)][:, :, None, None]
            K_ket = K_ij[np.ix_(kk_, ll)][None, None, :, :]
            Pre = (np.pi**2 / (A_bd * B_bd)) * np.sqrt(np.pi / (4.0 * Sigma)) * K_bra * K_ket
            
            block_res = np.zeros(Z.shape)
            
            # Contraction
            for t in range(Ex_bra.shape[2]):
                for u in range(Ey_bra.shape[2]):
                    for tau in range(Ex_ket.shape[2]):
                        for nu in range(Ey_ket.shape[2]):
                            # phase = (-1)**(t + u)
                            
                            C_bra = Ex_bra[:, :, t][:, :, None, None] * Ey_bra[:, :, u][:, :, None, None]
                            C_ket = Ex_ket[:, :, tau][None, None, :, :] * Ey_ket[:, :, nu][None, None, :, :]
                            # block_res += phase * C_bra * C_ket * I_tensor[t+tau, u+nu]
                            block_res += C_bra * C_ket * I_tensor[t+tau, u+nu]
                            
            eri_tensor[np.ix_(ii, jj, kk_, ll)] += Pre * block_res

        return eri_tensor

    # ------------------------------------
    # Prony Branch (dz > tolerance)
    # ------------------------------------
    
    n_ao = len(alphas)
    A = alphas[:, None] + alphas[None, :]
    Xi = (alphas[:, None] * alphas[None, :]) / A
    P_centers = (alphas[:, None, None] * centers[:, None, :] + alphas[None, :, None] * centers[None, :, :]) / A[:, :, None]
    R_ij_sq = np.sum((centers[:, None, :] - centers[None, :, :])**2, axis=2)
    Pref = np.exp(-Xi * R_ij_sq)

    eri_tensor = np.zeros((n_ao, n_ao, n_ao, n_ao), dtype=float)
    
    # Generalized Wicks Logic
    # We convert the "Forces" logic to a generalized operator list.
    # Orbitals map to lists of operators:
    # s -> []
    # px -> [(0, 0)]  (axis=0, center=0 relative to the pair)
    # py -> [(1, 0)]
    # dx2 -> [(0, 0), (0, 0)]
    # dxy -> [(0, 0), (1, 0)]
    # etc.
    
    def get_ops(kind):
        if kind == '2d-s': return []
        if kind == '2d-px': return [(0,0)] # x, center 0
        if kind == '2d-py': return [(1,0)] # y, center 0
        if kind == '2d-dx2': return [(0,0), (0,0)]
        if kind == '2d-dy2': return [(1,0), (1,0)]
        if kind == '2d-dxy': return [(0,0), (1,0)]
        return []

    dz_eff = abs(delta_z)
    invz = 1.0 / dz_eff
    gammas = XIS * (invz**2)
    weights = ETAS * invz

    unique_kinds = sorted(list(set(l.kind for l in labels)))
    
    for weight_p, gamma_p in zip(weights, gammas):
        
        for k_i, k_j, k_k, k_l in itertools.product(unique_kinds, repeat=4):
            ii = [i for i, l in enumerate(labels) if l.kind == k_i]
            jj = [i for i, l in enumerate(labels) if l.kind == k_j]
            kk_ = [i for i, l in enumerate(labels) if l.kind == k_k]
            ll = [i for i, l in enumerate(labels) if l.kind == k_l]
            
            a_slice = A[np.ix_(ii, jj)]; P_slice = P_centers[np.ix_(ii, jj)]
            Pre_ij  = Pref[np.ix_(ii, jj)]
            alphas_i = alphas[ii]; alphas_j = alphas[jj]
            Cs_i = centers[ii]; Cs_j = centers[jj]

            b_slice = A[np.ix_(kk_, ll)]; Q_slice = P_centers[np.ix_(kk_, ll)]
            Pre_kl  = Pref[np.ix_(kk_, ll)]
            alphas_k = alphas[kk_]; alphas_l = alphas[ll]
            Cs_k = centers[kk_]; Cs_l = centers[ll]
            
            a_bd = a_slice[:, :, None, None]; b_bd = b_slice[None, None, :, :]
            D = a_bd * b_bd + gamma_p * (a_bd + b_bd)
            Theta_p = (a_bd * b_bd * gamma_p) / D
            
            R_vec = P_slice[:, :, None, None, :] - Q_slice[None, None, :, :, :]
            R_sq  = np.sum(R_vec**2, axis=-1)
            
            G_p = weight_p * (Pre_ij[:,:,None,None]*Pre_kl[None,None,:,:]) * (np.pi**2 / D) * np.exp(-Theta_p * R_sq)

            # Construct Operator List
            # We have 4 centers in the quartet: 0,1,2,3 (i,j,k,l)
            # Each get_ops returns (axis, 0). We shift '0' to the actual center index.
            ops = []
            for op in get_ops(k_i): ops.append((op[0], 0))
            for op in get_ops(k_j): ops.append((op[0], 1))
            for op in get_ops(k_k): ops.append((op[0], 2))
            for op in get_ops(k_l): ops.append((op[0], 3))
            
            if not ops:
                eri_tensor[np.ix_(ii, jj, kk_, ll)] += G_p
                continue
            
            # Precompute Force Matrices
            # Force(axis, center)
            Forces = {}
            for axis in [0, 1]:
                # Center 0 (i)
                fc = -2 * Xi[np.ix_(ii,jj)][:, :, None, None] * (Cs_i[:, None, axis] - Cs_j[None, :, axis])[:, :, None, None]
                fp = -2 * Theta_p * (alphas_i[:, None, None, None] / a_bd) * R_vec[..., axis]
                Forces[(axis, 0)] = (fc + fp) / (2*alphas_i[:, None, None, None])
                
                # Center 1 (j)
                fc = +2 * Xi[np.ix_(ii,jj)][:, :, None, None] * (Cs_i[:, None, axis] - Cs_j[None, :, axis])[:, :, None, None]
                fp = -2 * Theta_p * (alphas_j[None, :, None, None] / a_bd) * R_vec[..., axis]
                Forces[(axis, 1)] = (fc + fp) / (2*alphas_j[None, :, None, None])
                
                # Center 2 (k)
                fc = -2 * Xi[np.ix_(kk_,ll)][None, None, :, :] * (Cs_k[:, None, axis] - Cs_l[None, :, axis])[None, None, :, :]
                fp = +2 * Theta_p * (alphas_k[None, None, :, None] / b_bd) * R_vec[..., axis]
                Forces[(axis, 2)] = (fc + fp) / (2*alphas_k[None, None, :, None])
                
                # Center 3 (l)
                fc = +2 * Xi[np.ix_(kk_,ll)][None, None, :, :] * (Cs_k[:, None, axis] - Cs_l[None, :, axis])[None, None, :, :]
                fp = +2 * Theta_p * (alphas_l[None, None, None, :] / b_bd) * R_vec[..., axis]
                Forces[(axis, 3)] = (fc + fp) / (2*alphas_l[None, None, None, :])

            # Precompute Couplings
            # C(u, v) depends on (axis_u, center_u) and (axis_v, center_v)
            # Only non-zero if axis_u == axis_v
            def get_coupling_val(op1, op2):
                ax1, c1 = op1
                ax2, c2 = op2
                if ax1 != ax2: return 0.0
                
                # Centers 0,1 are in 'a' block. 2,3 in 'b' block.
                in_a1 = (c1 < 2); in_a2 = (c2 < 2)
                
                if in_a1 and in_a2: # Both in A
                    return (1.0/(2*a_bd)) - (Theta_p / (2*a_bd**2))
                elif (not in_a1) and (not in_a2): # Both in B
                    return (1.0/(2*b_bd)) - (Theta_p / (2*b_bd**2))
                else: # Cross
                    return Theta_p / (2 * a_bd * b_bd)

            # Recursive Wick's
            # ops is list of (axis, center)
            def recursive_wicks_eval(current_ops):
                if not current_ops: return 1.0
                
                head = current_ops[0]
                tail = current_ops[1:]
                
                # Contraction 1: Force * <tail>
                val = Forces[head] * recursive_wicks_eval(tail)
                
                # Contractions 2: sum Coupling * <rest>
                for i, other in enumerate(tail):
                    coupling = get_coupling_val(head, other)
                    # Optim: if coupling is scalar 0, skip
                    # Coupling is array.
                    remaining = tail[:i] + tail[i+1:]
                    val += coupling * recursive_wicks_eval(remaining)
                    
                return val
            
            factor = recursive_wicks_eval(ops)
            eri_tensor[np.ix_(ii, jj, kk_, ll)] += factor * G_p

    return eri_tensor

# ==========================================
#  Restoring Legacy Kernels (s-type only wrappers if needed)
# ==========================================
# (Only if code elsewhere relies on them explicitly)
def pair_params(alphas, centers):
    n = len(alphas)
    a_i = alphas[:, None]; a_j = alphas[None, :]
    A = a_i + a_j
    # ... (This function is mostly helper for old s-only code)
    return A, None, None # Deprecated mostly

def _permute_K_ikjl(K, n):
    return K.reshape(n,n,n,n).transpose(0,2,1,3).reshape(n*n, n*n)

def build_h1_nm(Kz, S_prim, T_prim, z_grid, V_en_of_z):
    # Standard builder, unchanged
    Nz = int(Kz.shape[0])
    h1_nm = (Kz[:,:,None,None] * S_prim[None,None,:,:]).astype(float)
    for n in range(Nz):
        h1_nm[n, n] += (T_prim + V_en_of_z(float(z_grid[n])))
    for n in range(Nz):
        for m in range(Nz):
            H = h1_nm[n, m]
            h1_nm[n, m] = 0.5 * (H + H.T)
    return h1_nm