import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from scipy.special import i0e, i1e, ive

# ====================================================
#  Basis function labels
# ====================================================

@dataclass
class PrimitiveLabel:
    """
    Minimal label for one primitive GTO.
    kind: '2d-s', '2d-px', '2d-py', TODO: '3d-s', '3d-px', '2d-dxy'...
    dim : 2 or 3          (2D slice GTO and TODO: full 3D GTO)
    l   : (lx, ly, lz)    Cartesian angular momentum exponents
    role: 'slice_2d' or 'global_3d'
    """
    kind: str
    dim: int
    l: Tuple[int, int, int]
    role: str = "slice_2d"


# ==================================
#  Prony & STO-6G data
# ==================================

ETAS = np.array([
    0.04173077, 0.0972349, 0.12927493, 0.13390385, 0.1212556,
    0.10187292, 0.0821291, 0.06475048, 0.05047605, 0.0391529,
    0.03033575, 0.02354503, 0.01836517, 0.0144772, 0.01165111,
    0.00969719, 0.00841568, 0.00761789, 0.00716047, 0.00695301
], dtype=np.float64)

XIS = np.array([
    2.90966766e+00, 1.46957062e+00, 8.13288156e-01, 4.63366477e-01,
    2.67827855e-01, 1.56348009e-01, 9.20302874e-02, 5.45820332e-02,
    3.26022522e-02, 1.96041226e-02, 1.18613296e-02, 7.21519437e-03,
    4.40506148e-03, 2.68870734e-03, 1.62584139e-03, 9.55255326e-04,
    5.25214288e-04, 2.50901875e-04, 8.67029335e-05, 9.44699339e-06
], dtype=np.float64)

STO6_EXPS_H = np.array(
    [35.52322122, 6.513143725, 1.822142904, 0.6259552659, 0.2430767471, 0.1001124280],
    dtype=float
)

Exp_631g_ss_H = np.array(
    [18.73113696, 2.825394365, 0.6401216923, 0.1612777588],
    dtype=float
)

def get_test_prony_coeffs(delta_z):
    return ETAS, XIS

# ===========================
#  Shared utilities (2D)
# ===========================

def pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    Ad = A[:, None, :] - B[None, :, :]
    return np.sum(Ad * Ad, axis=-1)

def _parse_2d_l(lbl: PrimitiveLabel) -> Tuple[int, int]:
    if lbl.dim != 2:
        raise ValueError("Only dim=2 primitives supported.")
    lx, ly, lz = lbl.l
    if lz != 0:
        raise ValueError("For 2D slice basis, lz must be 0.")
    return lx, ly

def make_xy_sp_primitive_basis(
    nuclei_tuples: List[Tuple[float, float, float, float]],
    exps_s: np.ndarray,
    exps_p: np.ndarray,
    decimals: int = 12,
):
    exps_s = np.asarray(exps_s, float).reshape(-1)
    exps_p = np.asarray(exps_p, float).reshape(-1)
    rows = [] 

    for (Z, x, y, z) in nuclei_tuples:
        x, y = float(x), float(y)
        for a in exps_s: rows.append((a, x, y, 0))
        for a in exps_p: rows.append((a, x, y, 1)) # px
        for a in exps_p: rows.append((a, x, y, 2)) # py

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
        if k == 0:
            labels.append(PrimitiveLabel(kind="2d-s",  dim=2, l=(0, 0, 0)))
        elif k == 1:
            labels.append(PrimitiveLabel(kind="2d-px", dim=2, l=(1, 0, 0)))
        elif k == 2:
            labels.append(PrimitiveLabel(kind="2d-py", dim=2, l=(0, 1, 0)))
    return alphas, centers, labels

def _unique_primitives_on_xy(nuclei_tuples, sto_exps, decimals=12):
    rows = []
    for (Z, x, y, z) in nuclei_tuples:
        for a in sto_exps:
            rows.append((float(a), float(x), float(y)))
    arr = np.asarray(rows, float)
    arr_r = np.round(arr, decimals=decimals)
    _, idx = np.unique(arr_r, axis=0, return_index=True)
    idx = np.sort(idx)
    return arr[idx, 0], arr[idx, 1:3]

def make_xy_primitives_on_nuclei(nuclei_tuples, decimals=12):
    return _unique_primitives_on_xy(nuclei_tuples, STO6_EXPS_H, decimals)

# ====================================
#  2D Overlap and Kinetic (Cartesian)
# ====================================

def overlap_2d_cartesian(alphas, centers, labels):
    alphas = np.asarray(alphas, float)
    centers = np.asarray(centers, float)
    N = alphas.shape[0]
    S = np.zeros((N, N), float)

    for i in range(N):
        aA = alphas[i]; xA, yA = centers[i]; lxi, lyi = _parse_2d_l(labels[i])
        for j in range(N):
            aB = alphas[j]; xB, yB = centers[j]; lxj, lyj = _parse_2d_l(labels[j])

            Rx = xA - xB; Ry = yA - yB; R2 = Rx*Rx + Ry*Ry
            gamma = aA + aB; mu = (aA * aB) / gamma
            S_ss  = np.pi / gamma * np.exp(-mu * R2)

            is_sA, is_pxA, is_pyA = (lxi==0 and lyi==0), (lxi==1), (lyi==1)
            is_sB, is_pxB, is_pyB = (lxj==0 and lyj==0), (lxj==1), (lyj==1)

            if is_sA and is_sB:     val = S_ss
            elif is_pxA and is_sB:  val = - (mu/aA)*Rx*S_ss
            elif is_pyA and is_sB:  val = - (mu/aA)*Ry*S_ss
            elif is_sA and is_pxB:  val = + (mu/aB)*Rx*S_ss
            elif is_sA and is_pyB:  val = + (mu/aB)*Ry*S_ss
            elif is_pxA and is_pxB: val = (mu/(2*aA*aB)) * S_ss * (1 - 2*mu*Rx*Rx)
            elif is_pyA and is_pyB: val = (mu/(2*aA*aB)) * S_ss * (1 - 2*mu*Ry*Ry)
            elif (is_pxA and is_pyB) or (is_pyA and is_pxB):
                val = - (mu*mu/(aA*aB)) * Rx * Ry * S_ss
            else: val = 0.0
            S[i, j] = val
    return 0.5 * (S + S.T)

def kinetic_2d_cartesian(alphas, centers, labels):
    alphas = np.asarray(alphas, float)
    centers = np.asarray(centers, float)
    N = alphas.shape[0]
    Tmat = np.zeros((N, N), float)

    for i in range(N):
        aA = alphas[i]; xA, yA = centers[i]; lxi, lyi = _parse_2d_l(labels[i])
        for j in range(N):
            aB = alphas[j]; xB, yB = centers[j]; lxj, lyj = _parse_2d_l(labels[j])
            Rx = xA - xB; Ry = yA - yB; R2 = Rx*Rx + Ry*Ry
            gamma = aA + aB; mu = (aA * aB) / gamma

            S_ss = np.pi / gamma * np.exp(-mu * R2)
            T_ss = S_ss * 2.0 * mu * (1.0 - mu * R2)
            Tprime  = 2.0 * mu**2 * S_ss * (mu * R2 - 2.0)
            Tsecond = 2.0 * mu**3 * S_ss * (3.0 - mu * R2)

            is_sA, is_pxA, is_pyA = (lxi==0 and lyi==0), (lxi==1), (lyi==1)
            is_sB, is_pxB, is_pyB = (lxj==0 and lyj==0), (lxj==1), (lyj==1)

            if is_sA and is_sB:     val = T_ss
            elif is_pxA and is_sB:  val = (Rx/aA) * Tprime
            elif is_pyA and is_sB:  val = (Ry/aA) * Tprime
            elif is_sA and is_pxB:  val = -(Rx/aB) * Tprime
            elif is_sA and is_pyB:  val = -(Ry/aB) * Tprime
            elif is_pxA and is_pxB: val = (1/(4*aA*aB)) * (-2*Tprime - 4*Rx*Rx*Tsecond)
            elif is_pyA and is_pyB: val = (1/(4*aA*aB)) * (-2*Tprime - 4*Ry*Ry*Tsecond)
            elif (is_pxA and is_pyB) or (is_pyA and is_pxB):
                val = - (Rx*Ry/(aA*aB)) * Tsecond
            else: val = 0.0
            Tmat[i, j] = val
    return 0.5 * (Tmat + Tmat.T)

# ==============================================
#  Electron-Nuclear Attraction 
# ==============================================

def _V_en_2d_cartesian_single_nucleus_dz0(alphas, centers, labels, nuc_xy):
    """Exact dz=0 e-n block. Fixed to remove unphysical exp(x) scaling."""
    n = len(alphas)
    out = np.zeros((n, n), dtype=float)
    nuc_x, nuc_y = float(nuc_xy[0]), float(nuc_xy[1])

    for i in range(n):
        ai = float(alphas[i]); Xi, Yi = float(centers[i,0]), float(centers[i,1])
        lxi, lyi = _parse_2d_l(labels[i])
        for j in range(n):
            aj = float(alphas[j]); Xj, Yj = float(centers[j,0]), float(centers[j,1])
            lxj, lyj = _parse_2d_l(labels[j])

            total_deg = lxi + lyi + lxj + lyj
            gamma = ai + aj; mu = (ai * aj) / gamma
            Px = (ai*Xi + aj*Xj)/gamma; Py = (ai*Yi + aj*Yj)/gamma
            Dx = Px - nuc_x; Dy = Py - nuc_y
            R2 = (Xi-Xj)**2 + (Yi-Yj)**2; D2 = Dx**2 + Dy**2

            x = 0.5 * gamma * D2
            i0, i1, i2 = i0e(x), i1e(x), ive(2, x)
            
            if i0 == 0.0: 
                out[i, j] = 0.0; continue

            R10 = i1 / i0; R20 = i2 / i0
            half_gamma = 0.5 * gamma
            
            L1 = half_gamma * (R10 - 1.0)
            L2 = (half_gamma ** 2) * (0.5 * (1.0 + R20) - R10 * R10)

            pref  = np.exp(-mu * R2)
            const = (np.pi ** 1.5) / np.sqrt(gamma)
            V_ss = pref * const * i0

            dR2_XA = 2*(Xi-Xj); dR2_XB = -2*(Xi-Xj); dR2_YA = 2*(Yi-Yj); dR2_YB = -2*(Yi-Yj)
            cA = ai/gamma; cB = aj/gamma
            dD2_XA = 2*cA*Dx; dD2_XB = 2*cB*Dx; dD2_YA = 2*cA*Dy; dD2_YB = 2*cB*Dy

            g_XA = -mu * dR2_XA + L1 * dD2_XA
            g_XB = -mu * dR2_XB + L1 * dD2_XB
            g_YA = -mu * dR2_YA + L1 * dD2_YA
            g_YB = -mu * dR2_YB + L1 * dD2_YB

            H_XAXB = -mu * (-2.0) + L2 * dD2_XA * dD2_XB + L1 * (2*cA*cB)
            H_YAYB = -mu * (-2.0) + L2 * dD2_YA * dD2_YB + L1 * (2*cA*cB)
            H_XAYB = L2 * dD2_XA * dD2_YB
            H_YAXB = L2 * dD2_YA * dD2_XB

            is_sA, is_pxA, is_pyA = (lxi==0 and lyi==0), (lxi==1), (lyi==1)
            is_sB, is_pxB, is_pyB = (lxj==0 and lyj==0), (lxj==1), (lyj==1)

            if total_deg == 0:
                val = V_ss
            elif total_deg == 1:
                dV_XA = V_ss * g_XA; dV_YA = V_ss * g_YA
                dV_XB = V_ss * g_XB; dV_YB = V_ss * g_YB
                if is_pxA and is_sB:   val = (1/(2*ai)) * dV_XA
                elif is_pyA and is_sB: val = (1/(2*ai)) * dV_YA
                elif is_sA and is_pxB: val = (1/(2*aj)) * dV_XB
                elif is_sA and is_pyB: val = (1/(2*aj)) * dV_YB
                else: val = 0.0
            elif total_deg == 2:
                d2V_XAXB = V_ss * (g_XA * g_XB + H_XAXB)
                d2V_YAYB = V_ss * (g_YA * g_YB + H_YAYB)
                d2V_XAYB = V_ss * (g_XA * g_YB + H_XAYB)
                d2V_YAXB = V_ss * (g_YA * g_XB + H_YAXB)
                scale = 1.0 / (4.0 * ai * aj)
                if is_pxA and is_pxB:   val = scale * d2V_XAXB
                elif is_pyA and is_pyB: val = scale * d2V_YAYB
                elif is_pxA and is_pyB: val = scale * d2V_XAYB
                elif is_pyA and is_pxB: val = scale * d2V_YAXB
                else: val = 0.0
            else: val = 0.0
            out[i, j] = val
    return 0.5 * (out + out.T)

def V_en_2d_cartesian_single_nucleus(alphas, centers, labels, nuc_xy, dz_abs):
    """Prony branch for dz > 0."""
    if abs(dz_abs) <= 1e-6:
        return _V_en_2d_cartesian_single_nucleus_dz0(alphas, centers, labels, nuc_xy)
    
    invz = 1.0 / abs(dz_abs)
    n = len(alphas)
    out = np.zeros((n, n), dtype=float)

    for i in range(n):
        ai = alphas[i]; Xi, Yi = centers[i]; lxi, lyi = _parse_2d_l(labels[i])
        for j in range(n):
            aj = alphas[j]; Xj, Yj = centers[j]; lxj, lyj = _parse_2d_l(labels[j])
            
            gamma = ai+aj; mu = ai*aj/gamma
            Px = (ai*Xi+aj*Xj)/gamma; Py = (ai*Yi+aj*Yj)/gamma
            dPx = Px - nuc_xy[0]; dPy = Py - nuc_xy[1]
            R2 = (Xi-Xj)**2 + (Yi-Yj)**2; D2 = dPx**2 + dPy**2
            cA = ai/gamma; cB = aj/gamma

            # Accumulate Prony terms
            V_ss = 0.0
            dV_XA = 0.0; dV_YA = 0.0; dV_XB = 0.0; dV_YB = 0.0
            GP_XAXB = 0.0; GP_YAYB = 0.0; GP_XAYB = 0.0; GP_YAXB = 0.0
            H_XAXB_sum = 0.0; H_YAYB_sum = 0.0

            for eta, xi in zip(ETAS, XIS):
                gam_p = xi * invz**2
                lam = gamma * gam_p / (gamma + gam_p)
                C = (eta * invz) * (np.pi / (gamma + gam_p))
                E = -mu * R2 - lam * D2
                term = C * np.exp(E)
                
                V_ss += term
                
                dE_XA = -2*mu*(Xi-Xj) - 2*lam*cA*dPx
                dE_XB = +2*mu*(Xi-Xj) - 2*lam*cB*dPx
                dE_YA = -2*mu*(Yi-Yj) - 2*lam*cA*dPy
                dE_YB = +2*mu*(Yi-Yj) - 2*lam*cB*dPy
                
                dV_XA += term * dE_XA; dV_YA += term * dE_YA
                dV_XB += term * dE_XB; dV_YB += term * dE_YB

                Hxx = 2*mu - 2*lam*(ai*aj/gamma**2)
                
                GP_XAXB += term * dE_XA * dE_XB; GP_YAYB += term * dE_YA * dE_YB
                GP_XAYB += term * dE_XA * dE_YB; GP_YAXB += term * dE_YA * dE_XB
                H_XAXB_sum += term * Hxx; H_YAYB_sum += term * Hxx

            total_deg = lxi+lyi+lxj+lyj
            if total_deg == 0: val = V_ss
            elif total_deg == 1:
                if lxi==1: val = dV_XA/(2*ai)
                elif lyi==1: val = dV_YA/(2*ai)
                elif lxj==1: val = dV_XB/(2*aj)
                elif lyj==1: val = dV_YB/(2*aj)
                else: val = 0.0
            elif total_deg == 2:
                scale = 1.0/(4*ai*aj)
                if lxi==1 and lxj==1: val = scale*(GP_XAXB + H_XAXB_sum)
                elif lyi==1 and lyj==1: val = scale*(GP_YAYB + H_YAYB_sum)
                elif lxi==1 and lyj==1: val = scale*GP_XAYB
                elif lyi==1 and lxj==1: val = scale*GP_YAXB
                else: val = 0.0
            else: val = 0.0
            out[i, j] = val
    return 0.5 * (out + out.T)

def V_en_sp_total_at_z(alphas, centers, labels, nuclei_tuples, z):
    N = len(alphas)
    V = np.zeros((N, N), float)
    for (Z, xA, yA, zA) in nuclei_tuples:
        V -= Z * V_en_2d_cartesian_single_nucleus(alphas, centers, labels, np.array([xA,yA]), abs(z-zA))
    return V

# ==========================================
#  RESTORED LEGACY KERNELS (For s-type M=1)
# ==========================================

def pair_params(alphas, centers):
    """Shared AO pair parameters (s-type)."""
    n = len(alphas)
    a_i = alphas[:, None]; a_j = alphas[None, :]
    A = a_i + a_j
    Ai = centers[:, None, :]; Aj = centers[None, :, :]
    P = (a_i[..., None] * Ai + a_j[..., None] * Aj) / A[..., None]
    Rij2 = pairwise_sqdist(centers, centers)
    pref = np.exp(-(a_i * a_j / A) * Rij2)
    return A, P, pref

def pair_kernel_matrix_gamma(a_sum, Pij, pref, gamma):
    n = a_sum.shape[0]; n2 = n*n
    α = a_sum.reshape(n2); X0 = Pij[...,0].reshape(n2); Y0 = Pij[...,1].reshape(n2); pre = pref.reshape(n2)
    αp = α[:,None]; αpp = α[None,:]
    
    def get_comp(u0, u0p):
        Ax = αp + gamma; Bx = αpp + gamma
        detMx = Ax*Bx - gamma**2
        inv11 = Bx/detMx; inv12 = gamma/detMx; inv22 = Ax/detMx
        q1 = αp*u0; q2 = αpp*u0p
        qMq = inv11*q1**2 + 2*inv12*q1*q2 + inv22*q2**2
        return (np.pi/np.sqrt(detMx)) * np.exp(qMq - αp*u0**2 - αpp*u0p**2)

    Kx = get_comp(X0[:,None], X0[None,:])
    Ky = get_comp(Y0[:,None], Y0[None,:])
    return (pre[:,None]*pre[None,:]) * Kx * Ky

def pair_kernel_matrix_exact_dz0(a_sum, Pij, pref):
    n = a_sum.shape[0]; n2 = n*n
    a_flat = a_sum.reshape(n2); P_flat = Pij.reshape(n2, 2); pref_fl = pref.reshape(n2)
    dP = P_flat[:,None,:] - P_flat[None,:,:]
    R2 = np.sum(dP**2, axis=2)
    a_col = a_flat[:,None]; b_row = a_flat[None,:]
    ab = a_col*b_row; apb = a_col+b_row
    kappa = ab/apb
    pref_prod = pref_fl[:,None] * pref_fl[None,:]
    const = (np.pi**2.5) / np.sqrt(ab*apb)
    x = 0.5 * kappa * R2
    return pref_prod * const * i0e(x)

def ao_K_of_delta(a_sum, Pij, pref, dz):
    if dz <= 1e-12:
        return pair_kernel_matrix_exact_dz0(a_sum, Pij, pref)
    invz = 1.0 / dz
    n2 = a_sum.shape[0]**2
    out = np.zeros((n2, n2), dtype=float)
    for eta, xi in zip(ETAS, XIS):
        out += (eta * invz) * pair_kernel_matrix_gamma(a_sum, Pij, pref, xi*invz**2)
    return out

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

# =========================================================
#  ERI (Two-Electron Integrals) - Pure 2D + Prony Dispatch
# =========================================================

def _hermite_coefficients_1d(dim_idx, i_idx, j_idx, alphas_i, alphas_j, centers_i, centers_j, A_ij, P_ij):
    ni, nj = len(alphas_i), len(alphas_j)
    shape = (ni, nj)
    X_PA = P_ij - centers_i[:, None]; X_PB = P_ij - centers_j[None, :]
    inv_2p = 1.0 / (2.0 * A_ij)

    E_ss = np.zeros(shape + (1,)); E_ss[..., 0] = 1.0
    E_sp = np.zeros(shape + (2,)); E_sp[..., 0] = X_PB; E_sp[..., 1] = inv_2p
    E_ps = np.zeros(shape + (2,)); E_ps[..., 0] = X_PA; E_ps[..., 1] = inv_2p
    E_pp = np.zeros(shape + (3,))
    E_pp[..., 0] = X_PB * E_ps[..., 0] + E_ps[..., 1]
    E_pp[..., 1] = inv_2p * E_ps[..., 0] + X_PB * E_ps[..., 1]
    E_pp[..., 2] = inv_2p * E_ps[..., 1]
    
    return E_ss, E_sp, E_ps, E_pp

def _bessel_recursion_2d(t_max, u_max, Z, Delta_x, Delta_y, p_eff):
    N_total = t_max + u_max
    n_batch = Z.shape[0]
    I_tensor = np.zeros((t_max + 1, u_max + 1, N_total + 1, n_batch))
    X = -Z
    vals_n = {}
    vals_n[0] = i0e(X); vals_n[1] = i1e(X)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        for n in range(1, N_total):
            term = (2 * n / X) * vals_n[n]
            term[X < 1e-10] = 0.0 
            vals_n[n+1] = vals_n[n-1] - term
            
    for n in range(N_total + 1):
        I_tensor[0, 0, n] = ((-1.0)**n) * vals_n[n]
        
    factor = -0.5 * p_eff 
    def get_n_sum(arr_slice, n_target):
        i_n = arr_slice[n_target]; i_np1 = arr_slice[n_target + 1]
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

def _eri_2d_pure_coulomb(alphas, centers, labels):
    import itertools
    n_ao = len(alphas)
    eri_tensor = np.zeros((n_ao, n_ao, n_ao, n_ao), dtype=float)
    A = alphas[:, None] + alphas[None, :]
    Xi = (alphas[:, None] * alphas[None, :]) / A
    P_centers = (alphas[:, None, None] * centers[:, None, :] + alphas[None, :, None] * centers[None, :, :]) / A[:, :, None]
    R_ij_sq = np.sum((centers[:, None, :] - centers[None, :, :])**2, axis=2)
    K_ij = np.exp(-Xi * R_ij_sq)
    
    type_map = {'2d-s': 0, '2d-px': 1, '2d-py': 2}
    indices_by_type = {0: [], 1: [], 2: []}
    for i, l in enumerate(labels): indices_by_type[type_map[l.kind]].append(i)
    active_types = [t for t in [0,1,2] if indices_by_type[t]]
    
    for ti, tj, tk, tl in itertools.product(active_types, repeat=4):
        ii = indices_by_type[ti]; jj = indices_by_type[tj]; kk = indices_by_type[tk]; ll = indices_by_type[tl]
        if not (ii and jj and kk and ll): continue
        
        A_bra = A[np.ix_(ii, jj)]; P_bra = P_centers[np.ix_(ii, jj)]
        Lix = 1 if ti == 1 else 0; Liy = 1 if ti == 2 else 0
        Ljx = 1 if tj == 1 else 0; Ljy = 1 if tj == 2 else 0
        Ex_bra_all = _hermite_coefficients_1d(0, Lix, Ljx, alphas[ii], alphas[jj], centers[ii, 0], centers[jj, 0], A_bra, P_bra[..., 0])
        Ey_bra_all = _hermite_coefficients_1d(1, Liy, Ljy, alphas[ii], alphas[jj], centers[ii, 1], centers[jj, 1], A_bra, P_bra[..., 1])
        idx_x_bra = (Lix << 1) | Ljx; idx_y_bra = (Liy << 1) | Ljy
        Ex_bra = Ex_bra_all[idx_x_bra if idx_x_bra < 3 else 3]; Ey_bra = Ey_bra_all[idx_y_bra if idx_y_bra < 3 else 3]
        
        B_ket = A[np.ix_(kk, ll)]; Q_ket = P_centers[np.ix_(kk, ll)]
        Lkx = 1 if tk == 1 else 0; Lky = 1 if tk == 2 else 0
        Llx = 1 if tl == 1 else 0; Lly = 1 if tl == 2 else 0
        Ex_ket_all = _hermite_coefficients_1d(0, Lkx, Llx, alphas[kk], alphas[ll], centers[kk, 0], centers[ll, 0], B_ket, Q_ket[..., 0])
        Ey_ket_all = _hermite_coefficients_1d(1, Lky, Lly, alphas[kk], alphas[ll], centers[kk, 1], centers[ll, 1], B_ket, Q_ket[..., 1])
        idx_x_ket = (Lkx << 1) | Llx
        idx_y_ket = (Lky << 1) | Lly
        Ex_ket = Ex_ket_all[idx_x_ket if idx_x_ket < 3 else 3]
        Ey_ket = Ey_ket_all[idx_y_ket if idx_y_ket < 3 else 3]
        
        P_bd = P_bra[:, :, None, None, :]
        Q_bd = Q_ket[None, None, :, :, :]
        Delta = Q_bd - P_bd # Delta = Q - P
        A_bd = A_bra[:, :, None, None]
        B_bd = B_ket[None, None, :, :]
        Sigma = (A_bd + B_bd) / (4.0 * A_bd * B_bd)
        Z = -np.sum(Delta**2, axis=-1) / (8.0 * Sigma)
        p_eff = 1.0 / (4.0 * Sigma)
        
        Tx_max = (Lix + Ljx) + (Lkx + Llx)
        Ty_max = (Liy + Ljy) + (Lky + Lly)
        batch_shape = Z.shape
        # Recurse with Delta (correct for Interaction derivatives wrt P with phase)
        I_tensor = _bessel_recursion_2d(Tx_max, Ty_max, Z.ravel(), Delta[..., 0].ravel(), Delta[..., 1].ravel(), p_eff.ravel())
        I_tensor = I_tensor.reshape((Tx_max+1, Ty_max+1) + batch_shape)
        
        K_bra = K_ij[np.ix_(ii, jj)][:, :, None, None]
        K_ket = K_ij[np.ix_(kk, ll)][None, None, :, :]
        Pre = (np.pi**2 / (A_bd * B_bd)) * np.sqrt(np.pi / (4.0 * Sigma)) * K_bra * K_ket
        block_res = np.zeros(batch_shape)
        
        for t in range(Ex_bra.shape[2]):
            for u in range(Ey_bra.shape[2]):
                for tau in range(Ex_ket.shape[2]):
                    for nu in range(Ey_ket.shape[2]):
                        phase = (-1)**(t + u)
                        C_bra = Ex_bra[:, :, t][:, :, None, None] * Ey_bra[:, :, u][:, :, None, None]
                        C_ket = Ex_ket[:, :, tau][None, None, :, :] * Ey_ket[:, :, nu][None, None, :, :]
                        block_res += phase * C_bra * C_ket * I_tensor[t+tau, u+nu]
        eri_tensor[np.ix_(ii, jj, kk, ll)] += Pre * block_res
    return eri_tensor

def eri_2d_cartesian_with_p(alphas, centers, labels, delta_z, dz_tol=1.0e-2):
    if abs(delta_z) < dz_tol:
        return _eri_2d_pure_coulomb(alphas, centers, labels)

    import itertools
    n_ao = len(alphas)
    A = alphas[:, None] + alphas[None, :]
    Xi = (alphas[:, None] * alphas[None, :]) / A
    P_centers = (alphas[:, None, None] * centers[:, None, :] + alphas[None, :, None] * centers[None, :, :]) / A[:, :, None]
    R_ij_sq = np.sum((centers[:, None, :] - centers[None, :, :])**2, axis=2)
    Pref = np.exp(-Xi * R_ij_sq)

    eri_tensor = np.zeros((n_ao, n_ao, n_ao, n_ao), dtype=float)
    kind_map = {'2d-s': None, '2d-px': 0, '2d-py': 1}
    
    dz_eff = abs(delta_z)
    invz = 1.0 / dz_eff
    gammas = XIS * (invz**2)
    weights = ETAS * invz

    for weight_p, gamma_p in zip(weights, gammas):
        unique_kinds = sorted(list(set(l.kind for l in labels)))
        
        for k_i, k_j, k_k, k_l in itertools.product(unique_kinds, repeat=4):
            ii = [i for i, l in enumerate(labels) if l.kind == k_i]
            jj = [i for i, l in enumerate(labels) if l.kind == k_j]
            kk = [i for i, l in enumerate(labels) if l.kind == k_k]
            ll = [i for i, l in enumerate(labels) if l.kind == k_l]
            if not (ii and jj and kk and ll): continue
            
            a_slice = A[np.ix_(ii, jj)]; P_slice = P_centers[np.ix_(ii, jj)]
            Xi_slice_ij = Xi[np.ix_(ii, jj)]; Pre_ij  = Pref[np.ix_(ii, jj)]
            alphas_i = alphas[ii]; alphas_j = alphas[jj]; Cs_i = centers[ii]; Cs_j = centers[jj]

            b_slice = A[np.ix_(kk, ll)]; Q_slice = P_centers[np.ix_(kk, ll)]
            Xi_slice_kl = Xi[np.ix_(kk, ll)]; Pre_kl  = Pref[np.ix_(kk, ll)]
            alphas_k = alphas[kk]; alphas_l = alphas[ll]; Cs_k = centers[kk]; Cs_l = centers[ll]

            a_bd = a_slice[:, :, None, None]; b_bd = b_slice[None, None, :, :]
            D = a_bd * b_bd + gamma_p * (a_bd + b_bd)
            Theta_p = (a_bd * b_bd * gamma_p) / D
            R_vec = P_slice[:, :, None, None, :] - Q_slice[None, None, :, :, :]
            R_sq  = np.sum(R_vec**2, axis=-1)
            
            G_p = weight_p * (Pre_ij[:,:,None,None]*Pre_kl[None,None,:,:]) * (np.pi**2 / D) * np.exp(-Theta_p * R_sq)

            p_ops = []
            if kind_map[k_i] is not None: p_ops.append((0, kind_map[k_i]))
            if kind_map[k_j] is not None: p_ops.append((1, kind_map[k_j]))
            if kind_map[k_k] is not None: p_ops.append((2, kind_map[k_k]))
            if kind_map[k_l] is not None: p_ops.append((3, kind_map[k_l]))
            
            if not p_ops:
                eri_tensor[np.ix_(ii, jj, kk, ll)] += G_p
                continue

            Forces = {}
            def get_force(pos, axis):
                if pos == 0:
                    fc = -2 * Xi_slice_ij[:, :, None, None] * (Cs_i[:, None, axis] - Cs_j[None, :, axis])[:, :, None, None]
                    fp = -2 * Theta_p * (alphas_i[:, None, None, None] / a_bd) * R_vec[..., axis]
                    return (fc + fp) / (2*alphas_i[:, None, None, None])
                elif pos == 1:
                    fc = +2 * Xi_slice_ij[:, :, None, None] * (Cs_i[:, None, axis] - Cs_j[None, :, axis])[:, :, None, None]
                    fp = -2 * Theta_p * (alphas_j[None, :, None, None] / a_bd) * R_vec[..., axis]
                    return (fc + fp) / (2*alphas_j[None, :, None, None])
                elif pos == 2:
                    fc = -2 * Xi_slice_kl[None, None, :, :] * (Cs_k[:, None, axis] - Cs_l[None, :, axis])[None, None, :, :]
                    fp = +2 * Theta_p * (alphas_k[None, None, :, None] / b_bd) * R_vec[..., axis]
                    return (fc + fp) / (2*alphas_k[None, None, :, None])
                elif pos == 3:
                    fc = +2 * Xi_slice_kl[None, None, :, :] * (Cs_k[:, None, axis] - Cs_l[None, :, axis])[None, None, :, :]
                    fp = +2 * Theta_p * (alphas_l[None, None, None, :] / b_bd) * R_vec[..., axis]
                    return (fc + fp) / (2*alphas_l[None, None, None, :])

            for item in p_ops: Forces[item] = get_force(*item)

            def get_coupling(u, v):
                if u > v: u, v = v, u
                if u==0 and v==1: return (1.0/(2*a_bd)) - (Theta_p / (2*a_bd**2))
                if u==2 and v==3: return (1.0/(2*b_bd)) - (Theta_p / (2*b_bd**2))
                return Theta_p / (2 * a_bd * b_bd)

            def recursive_wicks(ops):
                if not ops: return 1.0
                head = ops[0]; tail = ops[1:]
                val = Forces[head] * recursive_wicks(tail)
                for i, other in enumerate(tail):
                    if head[1] == other[1]:
                        val += get_coupling(head[0], other[0]) * recursive_wicks(tail[:i] + tail[i+1:])
                return val

            eri_tensor[np.ix_(ii, jj, kk, ll)] += recursive_wicks(p_ops) * G_p

    return eri_tensor