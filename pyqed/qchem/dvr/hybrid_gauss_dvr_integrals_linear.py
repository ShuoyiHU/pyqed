import numpy as np
from scipy.special import erfcx
from dataclasses import dataclass
from typing import Tuple, List

# ====================================================
#  Basis function labels
# ====================================================

@dataclass
class PrimitiveLabel:
    kind: str
    dim: int
    l: Tuple[int, int, int]
    role: str = "slice_2d"

# =========================================================
#  Basis Construction (Forced Centered for Linear Case)
# =========================================================

def _parse_2d_l(lbl: PrimitiveLabel) -> Tuple[int, int]:
    return lbl.l[0], lbl.l[1]

def make_xy_spd_primitive_basis(
    nuclei_tuples: List[Tuple[float, float, float, float]],
    exps_s: np.ndarray,
    exps_p: np.ndarray,
    exps_d: np.ndarray = None,
    decimals: int = 12,
):
    """
    Generates primitive basis. 
    CRITICAL: For this analytical module, we assume all centers are at (0,0).
    """
    exps_s = np.asarray(exps_s, float).reshape(-1)
    exps_p = np.asarray(exps_p, float).reshape(-1)
    if exps_d is None: exps_d = np.array([], float)
    else: exps_d = np.asarray(exps_d, float).reshape(-1)
    
    rows = []
    # Force centers to 0.0 for analytical linear chain overlap/orthogonality
    x, y = 0.0, 0.0 
    
    # s
    for a in exps_s: rows.append((a, x, y, 0))
    # p
    for a in exps_p: rows.append((a, x, y, 1)) # px
    for a in exps_p: rows.append((a, x, y, 2)) # py
    # d
    for a in exps_d: rows.append((a, x, y, 3)) # dx2
    for a in exps_d: rows.append((a, x, y, 4)) # dy2
    for a in exps_d: rows.append((a, x, y, 5)) # dxy

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

# =========================================================
#  Standard 2D Overlap/Kinetic (Centered at 0,0)
# =========================================================

def _overlap_1d_centered(l1, l2, alpha, beta):
    """ Overlap of x^l1 exp(-ax^2) and x^l2 exp(-bx^2) centered at 0. """
    if (l1 + l2) % 2 != 0:
        return 0.0
    gamma = alpha + beta
    # Int x^k exp(-gamma x^2) = [(k-1)!! / (2 gamma)^(k/2)] * sqrt(pi/gamma)
    k = l1 + l2
    m = k // 2
    dfact = 1.0
    for x in range(1, 2*m, 2): dfact *= x
    return dfact / ((2*gamma)**m) * np.sqrt(np.pi / gamma)

def _kinetic_1d_centered(l1, l2, alpha, beta):
    """ Kinetic <1 | -0.5 d2/dx2 | 2> centered at 0. """
    # T = -0.5 [ 4b^2 S(l1, l2+2) - 2b(2l2+1) S(l1, l2) + l2(l2-1) S(l1, l2-2) ]
    term1 = 4.0 * beta**2 * _overlap_1d_centered(l1, l2 + 2, alpha, beta)
    term2 = -2.0 * beta * (2 * l2 + 1) * _overlap_1d_centered(l1, l2, alpha, beta)
    term3 = 0.0
    if l2 >= 2:
        term3 = l2 * (l2 - 1) * _overlap_1d_centered(l1, l2 - 2, alpha, beta)
    return -0.5 * (term1 + term2 + term3)

def overlap_2d_cartesian(alphas, centers, labels):
    # Ignores centers, assumes (0,0)
    N = len(alphas)
    S = np.zeros((N, N), float)
    for i in range(N):
        lxA, lyA = _parse_2d_l(labels[i])
        for j in range(i, N):
            lxB, lyB = _parse_2d_l(labels[j])
            Sx = _overlap_1d_centered(lxA, lxB, alphas[i], alphas[j])
            Sy = _overlap_1d_centered(lyA, lyB, alphas[i], alphas[j])
            val = Sx * Sy
            S[i, j] = val
            S[j, i] = val
    return S

def kinetic_2d_cartesian(alphas, centers, labels):
    # Ignores centers, assumes (0,0)
    N = len(alphas)
    Tmat = np.zeros((N, N), float)
    for i in range(N):
        lxA, lyA = _parse_2d_l(labels[i])
        for j in range(i, N):
            lxB, lyB = _parse_2d_l(labels[j])
            
            Sx = _overlap_1d_centered(lxA, lxB, alphas[i], alphas[j])
            Sy = _overlap_1d_centered(lyA, lyB, alphas[i], alphas[j])
            Tx = _kinetic_1d_centered(lxA, lxB, alphas[i], alphas[j])
            Ty = _kinetic_1d_centered(lyA, lyB, alphas[i], alphas[j])
            
            val = Tx * Sy + Sx * Ty
            Tmat[i, j] = val
            Tmat[j, i] = val
    return Tmat

# =========================================================
#  ANALYTICAL V_en (Electron-Nuclear Attraction)
# =========================================================

def V_en_analytical_linear(alphas, centers, labels, nuclei_tuples, z_grid_pt):
    """
    Computes V_en using the exact erfc formula for centered Gaussians.
    Strictly valid for linear molecules (Nuclei on Z-axis, Basis at X=Y=0).
    """
    N = len(alphas)
    V = np.zeros((N, N), float)
    
    for i in range(N):
        for j in range(i, N):
            gamma = alphas[i] + alphas[j]
            lxA, lyA = _parse_2d_l(labels[i])
            lxB, lyB = _parse_2d_l(labels[j])
            
            if (lxA + lxB) % 2 != 0 or (lyA + lyB) % 2 != 0:
                continue
            
            poly_prefactor = 1.0
            if (lxA+lxB) > 0 or (lyA+lyB) > 0:
                 # Standard Overlap Scaling for polynomial part approximation
                 moment_x = _overlap_1d_centered(lxA, lxB, alphas[i], alphas[j]) / np.sqrt(np.pi/gamma)
                 moment_y = _overlap_1d_centered(lyA, lyB, alphas[i], alphas[j]) / np.sqrt(np.pi/gamma)
                 poly_prefactor = moment_x * moment_y

            val = 0.0
            for (Z_nuc, xN, yN, zN) in nuclei_tuples:
                dz = abs(z_grid_pt - zN)
                U = np.sqrt(gamma) * dz
                
                # Correction: V = -Z * (pi * sqrt(pi) / sqrt(gamma)) * erfcx(U)
                # The integral Int exp(-g rho^2)/sqrt(rho^2+z^2) d2rho = pi^1.5/sqrt(g) * erfcx
                term = -Z_nuc * (np.pi * np.sqrt(np.pi) / np.sqrt(gamma)) * erfcx(U)
                val += term
            
            V[i, j] = val * poly_prefactor
            V[j, i] = V[i, j]
            
    return V

# =========================================================
#  ANALYTICAL ERI (Electron Repulsion)
# =========================================================

def eri_2d_analytical_linear(alphas, centers, labels, delta_z):
    """
    Computes ERI (i j | k l) using exact erfc formula for centered Gaussians.
    Strictly valid for linear case (centers at 0).
    """
    N = len(alphas)
    ERI = np.zeros((N, N, N, N), float)
    
    dz = abs(delta_z)
    G = alphas[:, None] + alphas[None, :]
    n_pair = N * N
    
    for ij in range(n_pair):
        i = ij // N
        j = ij % N
        gamma_A = G[i, j]
        
        lxA, lyA = _parse_2d_l(labels[i])
        lxB, lyB = _parse_2d_l(labels[j])
        if (lxA+lxB)%2 != 0 or (lyA+lyB)%2 != 0: continue
        
        poly_pre_A = 1.0
        if (lxA+lxB)>0 or (lyA+lyB)>0:
             mx = _overlap_1d_centered(lxA, lxB, alphas[i], alphas[j]) / np.sqrt(np.pi/gamma_A)
             my = _overlap_1d_centered(lyA, lyB, alphas[i], alphas[j]) / np.sqrt(np.pi/gamma_A)
             poly_pre_A = mx * my

        for kl in range(n_pair):
            k = kl // N
            l = kl % N
            gamma_B = G[k, l]
            
            lxC, lyC = _parse_2d_l(labels[k])
            lxD, lyD = _parse_2d_l(labels[l])
            if (lxC+lxD)%2 != 0 or (lyC+lyD)%2 != 0: continue
            
            poly_pre_B = 1.0
            if (lxC+lxD)>0 or (lyC+lyD)>0:
                 mx = _overlap_1d_centered(lxC, lxD, alphas[k], alphas[l]) / np.sqrt(np.pi/gamma_B)
                 my = _overlap_1d_centered(lyC, lyD, alphas[k], alphas[l]) / np.sqrt(np.pi/gamma_B)
                 poly_pre_B = mx * my
            
            Sigma = gamma_A + gamma_B
            Omega = (gamma_A * gamma_B) / Sigma
            Z_arg = np.sqrt(Omega) * dz
            
            # Correction: ERI = pi^2.5 / sqrt(gA * gB * Sigma) * erfcx
            # The factor of 2.0 was wrong (double counted normalization).
            # The denominator was gamma_A*gamma_B (dimension L^-4), should be sqrt(gA*gB) (dimension L^-2).
            
            prefactor = (np.pi**2.5) / np.sqrt(gamma_A * gamma_B * Sigma)
            
            val = prefactor * erfcx(Z_arg)
            ERI[i, j, k, l] = val * poly_pre_A * poly_pre_B

    return ERI

def build_h1_nm_analytical(Kz, S_prim, T_prim, z_grid, V_en_func):
    Nz = int(Kz.shape[0])
    h1_nm = (Kz[:,:,None,None] * S_prim[None,None,:,:]).astype(float)
    for n in range(Nz):
        V_z = V_en_func(float(z_grid[n]))
        h1_nm[n, n] += (T_prim + V_z)
        
    for n in range(Nz):
        for m in range(Nz):
            H = h1_nm[n, m]
            h1_nm[n, m] = 0.5 * (H + H.T)
    return h1_nm