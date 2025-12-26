import numpy as np
import itertools
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from scipy.special import ive, i0e
from scipy.interpolate import CubicSpline
from scipy.integrate import dblquad, IntegrationWarning

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

ETAS = np.array([
    0.30069033, 0.16182767, 0.12407565, 0.09663122, 0.07452011,
    0.056974, 0.0433538, 0.03294897, 0.02506173, 0.0191048,
    0.01460944, 0.01121901, 0.00866935, 0.00677285, 0.00540124,
    0.00445656, 0.00384016, 0.00345875, 0.00324129, 0.00314306
], dtype=np.float64)

# Updated XIS values
XIS = np.array([
    8.95803564e-01, 3.76331786e-01, 1.99974776e-01, 1.11372688e-01,
    6.32311898e-02, 3.63237834e-02, 2.10614754e-02, 1.23135953e-02,
    7.25491440e-03, 4.30560406e-03, 2.57258719e-03, 1.54631802e-03,
    9.33542237e-04, 5.64018239e-04, 3.38088583e-04, 1.97286602e-04,
    1.07942037e-04, 5.14001243e-05, 1.77285290e-05, 1.92999376e-06
], dtype=np.float64)

# Standard Exp for H
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

def gamma_binom(n, k):
    # Simple nCk for small n
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
    
    Q1 = P - x1
    Q2 = P - x2
    
    val = 0.0
    for i in range(l1 + 1):
        comb1 = gamma_binom(l1, i)
        term1 = (Q1)**(l1 - i)
        for j in range(l2 + 1):
            comb2 = gamma_binom(l2, j)
            term2 = (Q2)**(l2 - j)
            
            k = i + j
            if k % 2 == 0:
                m = k // 2
                dfact = 1.0
                for x in range(1, 2*m, 2): dfact *= x
                
                gauss_int = dfact / ((2*gamma)**m)
                val += comb1 * comb2 * term1 * term2 * gauss_int
                
    return pre * val

def _kinetic_1d_explicit(l1, l2, x1, x2, alpha, beta):
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
            
            Sx = _overlap_1d_explicit(lxA, lxB, xA, xB, aA, aB)
            Sy = _overlap_1d_explicit(lyA, lyB, yA, yB, aA, aB)
            Tx = _kinetic_1d_explicit(lxA, lxB, xA, xB, aA, aB)
            Ty = _kinetic_1d_explicit(lyA, lyB, yA, yB, aA, aB)
            
            val = Tx * Sy + Sx * Ty
            Tmat[i, j] = val
            Tmat[j, i] = val
    return Tmat

# =============================================================
#  TABULATION CACHE SYSTEM (Fix for small dz instability)
# =============================================================

# Global cache variables
_VEN_SPLINES = {}   # Key: (nuc_index, i, j), Value: CubicSpline
_ERI_SPLINES = None # Key: None (stored as massive tensor of splines or dict), Value: Tensor of Splines or similar

# Config
TABULATION_CUTOFF = 0.40  # a.u.  Region where Prony fails/is inaccurate
TABULATION_POINTS = 10   # INCREASED for smoothness
TABULATION_GRID   = np.linspace(0, TABULATION_CUTOFF, TABULATION_POINTS)

def _exact_ven_integrand(y, x, xA, yA, lxA, lyA, aA, xB, yB, lxB, lyB, aB, xN, yN, dz_sq):
    """
    Integrand for V_en: phi_A(r) * phi_B(r) * (1 / sqrt((r-N)^2 + dz^2))
    """
    dist_sq = (x-xN)**2 + (y-yN)**2 + dz_sq
    
    # Singularity handling: If exactly on top of nucleus, return 0 (removable singularity for p/d)
    # or handle appropriately.
    # For numerical quadrature, we assume we don't hit 0 exactly unless forced.
    if dist_sq < 1e-13:
        # If we are here, we are at the nucleus.
        # For s-orbitals, this is divergent (1/0).
        # For p/d orbitals (lx+ly > 0), the numerator is 0, so limit is finite.
        # This function doesn't know L of inputs easily without checking lxA...
        # But we only use dblquad for dz=0 if NOT s-s.
        return 0.0

    # Gaussian A
    valA = (x-xA)**lxA * (y-yA)**lyA * np.exp(-aA*((x-xA)**2 + (y-yA)**2))
    # Gaussian B
    valB = (x-xB)**lxB * (y-yB)**lyB * np.exp(-aB*((x-xB)**2 + (y-yB)**2))
    
    op = 1.0 / np.sqrt(dist_sq)
    return valA * valB * op

def _generate_exact_ven_table_entry(alphas, centers, labels, nuclei_tuples):
    """
    Generates the spline table for V_en. 
    Uses Hybrid Approach:
    - dz=0, s-s shell: Analytical Bessel Formula.
    - dz=0, p/d shell: dblquad (singularity is removable).
    - dz>0: dblquad (smooth).
    """
    global _VEN_SPLINES
    if _VEN_SPLINES: return # Already initialized
    
    print(f"[Tabulation] Pre-computing V_en near-field (0 < dz < {TABULATION_CUTOFF}) for {len(alphas)} basis functions...")
    
    n_ao = len(alphas)
    
    for nuc_idx, (Z, xN, yN, zN) in enumerate(nuclei_tuples):
        for i in range(n_ao):
            for j in range(i, n_ao): # Symmetric
                
                # Parameters
                aA = alphas[i]; xA, yA = centers[i]; lxA, lyA = _parse_2d_l(labels[i])
                aB = alphas[j]; xB, yB = centers[j]; lxB, lyB = _parse_2d_l(labels[j])
                total_L = lxA + lyA + lxB + lyB
                
                # Integration bounds (approx 6 sigma)
                min_alpha = min(aA, aB)
                sigma = 1.0 / np.sqrt(min_alpha)
                bound = 6.0 * sigma
                gamma = aA + aB
                Px = (aA*xA + aB*xB)/gamma
                Py = (aA*yA + aB*yB)/gamma
                x_min, x_max = Px - bound, Px + bound
                y_min, y_max = Py - bound, Py + bound
                
                vals = []
                for dz in TABULATION_GRID:
                    # === Point dz = 0 ===
                    if dz < 1e-9:
                        if total_L == 0:
                            # Exact Analytical for s-s
                            K_AB = np.exp( -aA*aB/gamma * ((xA-xB)**2 + (yA-yB)**2) )
                            Dx = Px - xN; Dy = Py - yN
                            D2 = Dx**2 + Dy**2
                            x_arg = 0.5 * gamma * D2
                            val = K_AB * (np.pi / np.sqrt(gamma)) * np.sqrt(np.pi) * i0e(x_arg)
                            vals.append(val)
                            continue
                        else:
                            # dz=0 but p/d orbitals. Singularity is removable (numerator 0).
                            # dblquad handles this with the check in _exact_ven_integrand.
                            pass

                    # === Numerical Quadrature (dz > 0 OR dz=0 for p/d) ===
                    dz_sq = dz*dz
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=IntegrationWarning)
                        val, err = dblquad(
                            _exact_ven_integrand,
                            x_min, x_max,
                            lambda x: y_min, lambda x: y_max,
                            args=(xA, yA, lxA, lyA, aA, xB, yB, lxB, lyB, aB, xN, yN, dz_sq),
                            epsabs=1e-8, epsrel=1e-6
                        )
                    vals.append(val)
                # print('1')  #debug use
                vals = np.array(vals)
                # BC type: Enforce zero first derivative at dz=0 (symmetry)
                spline = CubicSpline(TABULATION_GRID, vals, bc_type=((1, 0.0), 'not-a-knot'))
                
                _VEN_SPLINES[(nuc_idx, i, j)] = spline
                _VEN_SPLINES[(nuc_idx, j, i)] = spline

    print(f"[Tabulation] V_en table built.")

def _generate_exact_eri_table_entry(alphas, centers, labels):
    """
    Generates spline table for ERI.
    Uses the Bessel-function based exact expansion (via `eri_2d_cartesian_with_p`'s small-dz branch)
    to generate the ground truth data.
    """
    global _ERI_SPLINES
    if _ERI_SPLINES is not None: return

    n_ao = len(alphas)
    print(f"[Tabulation] Pre-computing ERI near-field (0 < dz < {TABULATION_CUTOFF}). This may take a moment...")
    
    tensor_stack = []
    
    for k, dz in enumerate(TABULATION_GRID):
        # We perform the "exact" Bessel calculation
        # This function (defined later/below) has a branch for small dz
        val_tensor = eri_2d_cartesian_with_p(alphas, centers, labels, delta_z=dz, dz_tol=999.0)
        tensor_stack.append(val_tensor)
        
    tensor_stack = np.array(tensor_stack) # (Npts, N, N, N, N)
    
    print("[Tabulation] Fitting splines...")
    
    # FIX: Provide zero-derivative array of matching shape for bc_type
    zero_deriv = np.zeros((n_ao, n_ao, n_ao, n_ao), dtype=float)
    _ERI_SPLINES = CubicSpline(TABULATION_GRID, tensor_stack, axis=0, bc_type=((1, zero_deriv), 'not-a-knot'))
    print("[Tabulation] ERI table built.")


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
            
            gamma = aA + aB
            P_x = (aA * xA + aB * xB) / gamma
            P_y = (aA * yA + aB * yB) / gamma
            
            val_sum = 0.0
            for eta, xi in zip(ETAS, XIS):
                gam_p = xi * invz**2
                weight_p = eta * invz
                
                K_AB_x = np.exp(- aA*aB/gamma * (xA-xB)**2)
                K_AB_y = np.exp(- aA*aB/gamma * (yA-yB)**2)
                
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

def V_en_sp_total_at_z(alphas, centers, labels, nuclei_tuples, z):
    # Trigger Tabulation if empty
    if not _VEN_SPLINES:
        _generate_exact_ven_table_entry(alphas, centers, labels, nuclei_tuples)

    N = len(alphas)
    V = np.zeros((N, N), float)
    
    for nuc_idx, (Z, xA, yA, zA) in enumerate(nuclei_tuples):
        dz = abs(z - zA)
        
        if dz < TABULATION_CUTOFF:
            # === Use Tabulated Spline ===
            # Fill matrix from cache
            V_nuc = np.zeros((N, N), float)
            for i in range(N):
                for j in range(i, N):
                    spline = _VEN_SPLINES.get((nuc_idx, i, j))
                    if spline:
                        val = float(spline(dz))
                        V_nuc[i, j] = val
                        V_nuc[j, i] = val
        else:
            # === Use Prony ===
            # Note: The prony function does not handle the Z charge, it computes the integral.
            V_nuc = _V_en_prony_general(alphas, centers, labels, np.array([xA, yA]), dz)
            
        V -= Z * V_nuc
    return V

# =========================================================
#  ERI (Two-Electron Integrals) - Generalized
# =========================================================

def _hermite_coefficients_general(l_a, l_b, alpha_a, alpha_b, A, B):
    # (Unused in main path now, but kept for legacy or debugging if needed)
    pass

def _bessel_recursion_2d(t_max, u_max, Z, Delta_x, Delta_y, p_eff):
    """
    Stable recursion using scipy.special.ive to handle Z=0 singularity.
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


def eri_2d_cartesian_with_p(alphas, centers, labels, delta_z, dz_tol=None):
    """
    Generalized ERI with D-orbital support.
    
    Modes:
    1. dz_tol is None: Automatic Hybrid Mode (Spline < CUTOFF < Prony)
    2. dz_tol specified: Forces Exact Bessel logic if abs(dz) < dz_tol. 
       (Used for generating the table).
    """
    
    dz_eff = abs(delta_z)
    
    # === HYBRID SWITCHING LOGIC ===
    if dz_tol is None:
        # Standard Runtime Call
        
        # 1. Initialize Cache if needed
        if _ERI_SPLINES is None:
             _generate_exact_eri_table_entry(alphas, centers, labels)
             
        # 2. Check Cutoff
        if dz_eff < TABULATION_CUTOFF:
            # Use Spline Interpolation
            # _ERI_SPLINES is a CubicSpline object that computes vector output
            return _ERI_SPLINES(dz_eff)
            
        # 3. Else Fall through to Prony logic below (make sure to skip the exact block)
        use_exact_bessel = False
    else:
        # Generation/Forced Mode
        if dz_eff < dz_tol:
            use_exact_bessel = True
        else:
            use_exact_bessel = False

    # ------------------------------------
    # Exact Bessel Branch (for Tabulation)
    # ------------------------------------
    if use_exact_bessel:
        n_ao = len(alphas)
        eri_tensor = np.zeros((n_ao, n_ao, n_ao, n_ao), dtype=float)
        A = alphas[:, None] + alphas[None, :]
        Xi = (alphas[:, None] * alphas[None, :]) / A
        P_centers = (alphas[:, None, None] * centers[:, None, :] + alphas[None, :, None] * centers[None, :, :]) / A[:, :, None]
        R_ij_sq = np.sum((centers[:, None, :] - centers[None, :, :])**2, axis=2)
        K_ij = np.exp(-Xi * R_ij_sq)
        
        kind_to_L = {}
        for lbl in labels:
            kind_to_L[lbl.kind] = lbl.l # (lx, ly, lz)
            
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
                            t_vals = np.arange(1, Lmax+1)
                            H_tab[i,j,..., :-1] += t_vals * prev[..., 1:]
                            H_tab[i,j,..., 1:] += inv_2g[...,None] * prev[..., :-1]
                        else:
                            prev = H_tab[i, j-1]
                            X = (P_vec - cnt_b)
                            H_tab[i,j,...,:] += X[...,None] * prev
                            t_vals = np.arange(1, Lmax+1)
                            H_tab[i,j,..., :-1] += t_vals * prev[..., 1:]
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
            
            # dz term enters Z here:
            dz2 = dz_eff**2
            # Z = - ( |P-Q|^2 + dz^2 ) / (8 Sigma)
            R_xy_sq = np.sum(Delta**2, axis=-1)
            Z_arg = -(R_xy_sq + dz2) / (8.0 * Sigma)
            
            p_eff = 1.0 / (4.0 * Sigma)
            
            Tx_max = (Lix + Ljx) + (Lkx + Llx)
            Ty_max = (Liy + Ljy) + (Lky + Lly)
            
            # Call Bessel
            I_tensor = _bessel_recursion_2d(Tx_max, Ty_max, Z_arg.ravel(), Delta[..., 0].ravel(), Delta[..., 1].ravel(), p_eff.ravel())
            I_tensor = I_tensor.reshape((Tx_max+1, Ty_max+1) + Z_arg.shape)
            
            K_bra = K_ij[np.ix_(ii, jj)][:, :, None, None]
            K_ket = K_ij[np.ix_(kk_, ll)][None, None, :, :]
            Pre = (np.pi**2 / (A_bd * B_bd)) * np.sqrt(np.pi / (4.0 * Sigma)) * K_bra * K_ket
            
            block_res = np.zeros(Z_arg.shape)
            
            # Contraction
            for t in range(Ex_bra.shape[2]):
                for u in range(Ey_bra.shape[2]):
                    for tau in range(Ex_ket.shape[2]):
                        for nu in range(Ey_ket.shape[2]):
                            C_bra = Ex_bra[:, :, t][:, :, None, None] * Ey_bra[:, :, u][:, :, None, None]
                            C_ket = Ex_ket[:, :, tau][None, None, :, :] * Ey_ket[:, :, nu][None, None, :, :]
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
    
    def get_ops(kind):
        if kind == '2d-s': return []
        if kind == '2d-px': return [(0,0)]
        if kind == '2d-py': return [(1,0)]
        if kind == '2d-dx2': return [(0,0), (0,0)]
        if kind == '2d-dy2': return [(1,0), (1,0)]
        if kind == '2d-dxy': return [(0,0), (1,0)]
        return []

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

            ops = []
            for op in get_ops(k_i): ops.append((op[0], 0))
            for op in get_ops(k_j): ops.append((op[0], 1))
            for op in get_ops(k_k): ops.append((op[0], 2))
            for op in get_ops(k_l): ops.append((op[0], 3))
            
            if not ops:
                eri_tensor[np.ix_(ii, jj, kk_, ll)] += G_p
                continue
            
            Forces = {}
            for axis in [0, 1]:
                fc = -2 * Xi[np.ix_(ii,jj)][:, :, None, None] * (Cs_i[:, None, axis] - Cs_j[None, :, axis])[:, :, None, None]
                fp = -2 * Theta_p * (alphas_i[:, None, None, None] / a_bd) * R_vec[..., axis]
                Forces[(axis, 0)] = (fc + fp) / (2*alphas_i[:, None, None, None])
                
                fc = +2 * Xi[np.ix_(ii,jj)][:, :, None, None] * (Cs_i[:, None, axis] - Cs_j[None, :, axis])[:, :, None, None]
                fp = -2 * Theta_p * (alphas_j[None, :, None, None] / a_bd) * R_vec[..., axis]
                Forces[(axis, 1)] = (fc + fp) / (2*alphas_j[None, :, None, None])
                
                fc = -2 * Xi[np.ix_(kk_,ll)][None, None, :, :] * (Cs_k[:, None, axis] - Cs_l[None, :, axis])[None, None, :, :]
                fp = +2 * Theta_p * (alphas_k[None, None, :, None] / b_bd) * R_vec[..., axis]
                Forces[(axis, 2)] = (fc + fp) / (2*alphas_k[None, None, :, None])
                
                fc = +2 * Xi[np.ix_(kk_,ll)][None, None, :, :] * (Cs_k[:, None, axis] - Cs_l[None, :, axis])[None, None, :, :]
                fp = +2 * Theta_p * (alphas_l[None, None, None, :] / b_bd) * R_vec[..., axis]
                Forces[(axis, 3)] = (fc + fp) / (2*alphas_l[None, None, None, :])

            def get_coupling_val(op1, op2):
                ax1, c1 = op1
                ax2, c2 = op2
                if ax1 != ax2: return 0.0
                
                in_a1 = (c1 < 2); in_a2 = (c2 < 2)
                
                if in_a1 and in_a2:
                    return (1.0/(2*a_bd)) - (Theta_p / (2*a_bd**2))
                elif (not in_a1) and (not in_a2):
                    return (1.0/(2*b_bd)) - (Theta_p / (2*b_bd**2))
                else:
                    return Theta_p / (2 * a_bd * b_bd)

            def recursive_wicks_eval(current_ops):
                if not current_ops: return 1.0
                head = current_ops[0]
                tail = current_ops[1:]
                val = Forces[head] * recursive_wicks_eval(tail)
                for i, other in enumerate(tail):
                    coupling = get_coupling_val(head, other)
                    remaining = tail[:i] + tail[i+1:]
                    val += coupling * recursive_wicks_eval(remaining)
                return val
            
            factor = recursive_wicks_eval(ops)
            eri_tensor[np.ix_(ii, jj, kk_, ll)] += factor * G_p

    return eri_tensor

def pair_params(alphas, centers):
    n = len(alphas)
    a_i = alphas[:, None]; a_j = alphas[None, :]
    A = a_i + a_j
    return A, None, None

def _permute_K_ikjl(K, n):
    return K.reshape(n,n,n,n).transpose(0,2,1,3).reshape(n*n, n*n)

def build_h1_nm(Kz, S_prim, T_prim, z_grid, V_en_of_z):
    Nz = int(Kz.shape[0])
    h1_nm = (Kz[:,:,None,None] * S_prim[None,None,:,:]).astype(float)
    for n in range(Nz):
        h1_nm[n, n] += (T_prim + V_en_of_z(float(z_grid[n])))
    for n in range(Nz):
        for m in range(Nz):
            H = h1_nm[n, m]
            h1_nm[n, m] = 0.5 * (H + H.T)
    return h1_nm
