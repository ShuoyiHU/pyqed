import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Union
from scipy.special import i0e, i1e, ive, gamma

# ====================================================
#  Basis function labels
# ====================================================

@dataclass
class PrimitiveLabel:
    kind: str
    dim: int
    l: Tuple[int, int, int]
    role: str = "slice_2d"

@dataclass
class GTO3D:
    alpha: float
    center: np.ndarray # (x, y, z)
    l: Tuple[int, int, int] # (lx, ly, lz)
    norm: float = 1.0

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
    exps_s = np.asarray(exps_s, float).reshape(-1)
    exps_p = np.asarray(exps_p, float).reshape(-1)
    if exps_d is None: exps_d = np.array([], float)
    else: exps_d = np.asarray(exps_d, float).reshape(-1)
    
    rows = []
    for (Z, x, y, z) in nuclei_tuples:
        x, y = float(x), float(y)
        for a in exps_s: rows.append((a, x, y, 0))
        for a in exps_p: rows.append((a, x, y, 1)) 
        for a in exps_p: rows.append((a, x, y, 2)) 
        for a in exps_d: rows.append((a, x, y, 3))
        for a in exps_d: rows.append((a, x, y, 4))
        for a in exps_d: rows.append((a, x, y, 5))

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

make_xy_sp_primitive_basis = make_xy_spd_primitive_basis

# ====================================
#  General 1D Integral Helpers
# ====================================

def _overlap_1d(l1: int, l2: int, x1: float, x2: float, gamma: float) -> float:
    raise NotImplementedError("Use _overlap_1d_explicit")

def _overlap_1d_explicit(l1, l2, x1, x2, alpha, beta):
    gamma = alpha + beta
    P = (alpha * x1 + beta * x2) / gamma
    pre = np.exp(- (alpha * beta / gamma) * (x1 - x2)**2) * np.sqrt(np.pi / gamma)
    Q1 = P - x1; Q2 = P - x2
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

def gamma_binom(n, k):
    if k < 0 or k > n: return 0
    if k == 0 or k == n: return 1
    if k > n // 2: k = n - k
    res = 1
    for i in range(k):
        res = res * (n - i) // (i + 1)
    return res

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
        for j in range(i, N): 
            aB = alphas[j]; xB, yB = centers[j]; lxB, lyB = _parse_2d_l(labels[j])
            Sx = _overlap_1d_explicit(lxA, lxB, xA, xB, aA, aB)
            Sy = _overlap_1d_explicit(lyA, lyB, yA, yB, aA, aB)
            val = Sx * Sy
            S[i, j] = val; S[j, i] = val
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
            Tmat[i, j] = val; Tmat[j, i] = val
    return Tmat

# ==============================================
#  Electron-Nuclear Attraction (Generalized)
# ==============================================

def _V_en_prony_general(alphas, centers, labels, nuc_xy, dz_abs):
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
                    q1 = q - c1; q2 = q - c2
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
    N = len(alphas)
    V = np.zeros((N, N), float)
    for (Z, xA, yA, zA) in nuclei_tuples:
        dz = abs(z - zA)
        if dz < 1e-5: dz = 1e-5
        V_nuc = _V_en_prony_general(alphas, centers, labels, np.array([xA, yA]), dz)
        V -= Z * V_nuc
    return V

# =========================================================
#  ERI (Two-Electron Integrals) - Generalized
# =========================================================

def _bessel_recursion_2d(t_max, u_max, Z, Delta_x, Delta_y, p_eff):
    """
    Stable recursion using scipy.special.ive to handle Z=0 singularity.
    """
    N_total = t_max + u_max
    n_batch = Z.shape[0]
    vals_n = {}
    X = -Z
    for n in range(N_total + 1):
        vals_n[n] = ive(n, X)
    I_tensor = np.zeros((t_max + 1, u_max + 1, N_total + 1, n_batch))
    for n in range(N_total + 1):
        I_tensor[0, 0, n] = ((-1.0)**n) * vals_n[n]
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
    import itertools
    if abs(delta_z) < dz_tol:
        n_ao = len(alphas)
        eri_tensor = np.zeros((n_ao, n_ao, n_ao, n_ao), dtype=float)
        A = alphas[:, None] + alphas[None, :]
        Xi = (alphas[:, None] * alphas[None, :]) / A
        P_centers = (alphas[:, None, None] * centers[:, None, :] + alphas[None, :, None] * centers[None, :, :]) / A[:, :, None]
        R_ij_sq = np.sum((centers[:, None, :] - centers[None, :, :])**2, axis=2)
        K_ij = np.exp(-Xi * R_ij_sq)
        
        kind_to_L = {l.kind: l.l for l in labels}
        inds_by_kind = {}
        for i, lbl in enumerate(labels):
            if lbl.kind not in inds_by_kind: inds_by_kind[lbl.kind] = []
            inds_by_kind[lbl.kind].append(i)
        kinds = sorted(inds_by_kind.keys())
        
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

        for ki, kj, kk, kl in itertools.product(kinds, repeat=4):
            ii = inds_by_kind[ki]; jj = inds_by_kind[kj]
            kk_ = inds_by_kind[kk]; ll = inds_by_kind[kl]
            Lix, Liy = kind_to_L[ki][0:2]; Ljx, Ljy = kind_to_L[kj][0:2]
            Lkx, Lky = kind_to_L[kk][0:2]; Llx, Lly = kind_to_L[kl][0:2]
            
            A_bra = A[np.ix_(ii, jj)]; P_bra = P_centers[np.ix_(ii, jj)]
            B_ket = A[np.ix_(kk_, ll)]; Q_ket = P_centers[np.ix_(kk_, ll)]
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
            for t in range(Ex_bra.shape[2]):
                for u in range(Ey_bra.shape[2]):
                    for tau in range(Ex_ket.shape[2]):
                        for nu in range(Ey_ket.shape[2]):
                            phase = (-1)**(t + u)
                            C_bra = Ex_bra[:, :, t][:, :, None, None] * Ey_bra[:, :, u][:, :, None, None]
                            C_ket = Ex_ket[:, :, tau][None, None, :, :] * Ey_ket[:, :, nu][None, None, :, :]
                            block_res += phase * C_bra * C_ket * I_tensor[t+tau, u+nu]
            eri_tensor[np.ix_(ii, jj, kk_, ll)] += Pre * block_res
        return eri_tensor

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

    dz_eff = abs(delta_z)
    if dz_eff < 1e-5: dz_eff = 1e-5
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
                ax1, c1 = op1; ax2, c2 = op2
                if ax1 != ax2: return 0.0
                in_a1 = (c1 < 2); in_a2 = (c2 < 2)
                if in_a1 and in_a2: return (1.0/(2*a_bd)) - (Theta_p / (2*a_bd**2))
                elif (not in_a1) and (not in_a2): return (1.0/(2*b_bd)) - (Theta_p / (2*b_bd**2))
                else: return Theta_p / (2 * a_bd * b_bd)
            def recursive_wicks_eval(current_ops):
                if not current_ops: return 1.0
                head = current_ops[0]; tail = current_ops[1:]
                val = Forces[head] * recursive_wicks_eval(tail)
                for i, other in enumerate(tail):
                    coupling = get_coupling_val(head, other)
                    remaining = tail[:i] + tail[i+1:]
                    val += coupling * recursive_wicks_eval(remaining)
                return val
            factor = recursive_wicks_eval(ops)
            eri_tensor[np.ix_(ii, jj, kk_, ll)] += factor * G_p
    return eri_tensor

# =========================================================
#  New Mixed Integral Functions (3D GTO vs Hybrid)
# =========================================================

def evaluate_gto_z_on_grid(gto: GTO3D, z_grid: np.ndarray) -> np.ndarray:
    Az = gto.center[2]
    lz = gto.l[2]
    dz = z_grid - Az
    return (dz ** lz) * np.exp(-gto.alpha * (dz**2))

def overlap_mix_3d_hybrid(
    gtos: List[GTO3D],
    hybrids_alpha: np.ndarray,
    hybrids_center: np.ndarray,
    hybrids_label: List[PrimitiveLabel],
    z_grid: np.ndarray,
    dz: float
):
    # Correctly handle empty GTO list or empty hybrids
    if not gtos:
        n_hyb = len(hybrids_alpha)
        Nz = len(z_grid)
        return np.zeros((0, Nz * n_hyb))

    gto_alphas_2d = np.array([g.alpha for g in gtos])
    gto_centers_2d = np.array([g.center[:2] for g in gtos])
    gto_labels_2d = []
    for g in gtos:
        lx, ly = g.l[0], g.l[1]
        kind = '2d-s'
        if lx==1 and ly==0: kind = '2d-px'
        elif lx==0 and ly==1: kind = '2d-py'
        elif lx==2 and ly==0: kind = '2d-dx2'
        elif lx==0 and ly==2: kind = '2d-dy2'
        elif lx==1 and ly==1: kind = '2d-dxy'
        gto_labels_2d.append(PrimitiveLabel(kind=kind, dim=2, l=(lx, ly, 0)))
    
    n_gto = len(gtos)
    n_hyb = len(hybrids_alpha)
    Nz = len(z_grid)
    
    # Fix: Ensure gto_centers_2d is (N, 2)
    if len(gto_centers_2d) > 0 and gto_centers_2d.ndim == 1:
         # This shouldn't happen with list comp above, but safe guarding
         gto_centers_2d = gto_centers_2d.reshape(-1, 2)
         
    combined_alphas = np.concatenate([gto_alphas_2d, hybrids_alpha])
    combined_centers = np.concatenate([gto_centers_2d, hybrids_center])
    combined_labels = gto_labels_2d + hybrids_label
    
    S_all = overlap_2d_cartesian(combined_alphas, combined_centers, combined_labels)
    S_2D = S_all[:n_gto, n_gto:]
    
    chi_z_vals = np.zeros((n_gto, Nz))
    for i, g in enumerate(gtos):
        chi_z_vals[i, :] = evaluate_gto_z_on_grid(g, z_grid)
        
    S_mix = np.einsum('ms,mn->mns', S_2D, chi_z_vals) * np.sqrt(dz)
    return S_mix.reshape(n_gto, Nz * n_hyb)

def kinetic_mix_3d_hybrid(
    gtos: List[GTO3D],
    hybrids_alpha: np.ndarray,
    hybrids_center: np.ndarray,
    hybrids_label: List[PrimitiveLabel],
    z_grid: np.ndarray,
    T_dvr: np.ndarray, 
    dz: float
):
    if not gtos:
        n_hyb = len(hybrids_alpha)
        Nz = len(z_grid)
        return np.zeros((0, Nz * n_hyb))

    gto_alphas_2d = np.array([g.alpha for g in gtos])
    gto_centers_2d = np.array([g.center[:2] for g in gtos])
    gto_labels_2d = []
    for g in gtos:
        lx, ly = g.l[0], g.l[1]
        kind = '2d-s'
        if lx==1 and ly==0: kind = '2d-px'
        elif lx==0 and ly==1: kind = '2d-py'
        elif lx==2 and ly==0: kind = '2d-dx2'
        elif lx==0 and ly==2: kind = '2d-dy2'
        elif lx==1 and ly==1: kind = '2d-dxy'
        gto_labels_2d.append(PrimitiveLabel(kind=kind, dim=2, l=(lx, ly, 0)))

    n_gto = len(gtos)
    n_hyb = len(hybrids_alpha)
    Nz = len(z_grid)
    
    if len(gto_centers_2d) > 0:
        gto_centers_2d = gto_centers_2d.reshape(-1, 2)

    combined_alphas = np.concatenate([gto_alphas_2d, hybrids_alpha])
    combined_centers = np.concatenate([gto_centers_2d, hybrids_center])
    combined_labels = gto_labels_2d + hybrids_label
    
    S_all = overlap_2d_cartesian(combined_alphas, combined_centers, combined_labels)
    T_all = kinetic_2d_cartesian(combined_alphas, combined_centers, combined_labels)
    
    S_2D = S_all[:n_gto, n_gto:]
    T_2D = T_all[:n_gto, n_gto:]
    
    chi_z_vals = np.zeros((n_gto, Nz))
    for i, g in enumerate(gtos):
        chi_z_vals[i, :] = evaluate_gto_z_on_grid(g, z_grid)
        
    Term1 = np.einsum('ms,mn->mns', T_2D, chi_z_vals) * np.sqrt(dz)
    T_z_1d = np.einsum('mk,kn->mn', chi_z_vals, T_dvr) * np.sqrt(dz)
    Term2 = np.einsum('ms,mn->mns', S_2D, T_z_1d)
    
    T_mix = Term1 + Term2
    return T_mix.reshape(n_gto, Nz * n_hyb)

def ven_mix_3d_hybrid(
    gtos: List[GTO3D],
    hybrids_alpha: np.ndarray,
    hybrids_center: np.ndarray,
    hybrids_label: List[PrimitiveLabel],
    z_grid: np.ndarray,
    nuclei_tuples: List[Tuple[float, float, float, float]],
    dz: float
):
    if not gtos:
        n_hyb = len(hybrids_alpha)
        Nz = len(z_grid)
        return np.zeros((0, Nz * n_hyb))

    n_gto = len(gtos)
    n_hyb = len(hybrids_alpha)
    Nz = len(z_grid)
    
    gto_alphas_2d = np.array([g.alpha for g in gtos])
    gto_centers_2d = np.array([g.center[:2] for g in gtos])
    gto_labels_2d = []
    for g in gtos:
        lx, ly = g.l[0], g.l[1]
        kind = '2d-s'
        if lx==1 and ly==0: kind = '2d-px'
        elif lx==0 and ly==1: kind = '2d-py'
        elif lx==2 and ly==0: kind = '2d-dx2'
        elif lx==0 and ly==2: kind = '2d-dy2'
        elif lx==1 and ly==1: kind = '2d-dxy'
        gto_labels_2d.append(PrimitiveLabel(kind=kind, dim=2, l=(lx, ly, 0)))
        
    if len(gto_centers_2d) > 0:
        gto_centers_2d = gto_centers_2d.reshape(-1, 2)

    combined_alphas = np.concatenate([gto_alphas_2d, hybrids_alpha])
    combined_centers = np.concatenate([gto_centers_2d, hybrids_center])
    combined_labels = gto_labels_2d + hybrids_label
    
    chi_z_vals = np.zeros((n_gto, Nz))
    for i, g in enumerate(gtos):
        chi_z_vals[i, :] = evaluate_gto_z_on_grid(g, z_grid)
        
    V_mix = np.zeros((n_gto, Nz, n_hyb))
    for n in range(Nz):
        zn = z_grid[n]
        V_2D_all = V_en_sp_total_at_z(combined_alphas, combined_centers, combined_labels, nuclei_tuples, zn)
        V_2D_block = V_2D_all[:n_gto, n_gto:] 
        V_mix[:, n, :] = V_2D_block * chi_z_vals[:, n:n+1] * np.sqrt(dz)
        
    return V_mix.reshape(n_gto, Nz * n_hyb)

def eri_mix_3d_hybrid(
    gtos: List[GTO3D],
    hybrids_alpha: np.ndarray,
    hybrids_center: np.ndarray,
    hybrids_label: List[PrimitiveLabel],
    z_grid: np.ndarray,
    dz: float
):
    if not gtos:
        n_hyb = len(hybrids_alpha)
        Nz = len(z_grid)
        return {}, np.zeros((0, Nz))

    gto_alphas_2d = np.array([g.alpha for g in gtos])
    gto_centers_2d = np.array([g.center[:2] for g in gtos])
    gto_labels_2d = []
    for g in gtos:
        lx, ly = g.l[0], g.l[1]
        kind = '2d-s'
        if lx==1 and ly==0: kind = '2d-px'
        elif lx==0 and ly==1: kind = '2d-py'
        elif lx==2 and ly==0: kind = '2d-dx2'
        elif lx==0 and ly==2: kind = '2d-dy2'
        elif lx==1 and ly==1: kind = '2d-dxy'
        gto_labels_2d.append(PrimitiveLabel(kind=kind, dim=2, l=(lx, ly, 0)))
        
    n_gto = len(gtos)
    n_hyb = len(hybrids_alpha)
    Nz = len(z_grid)
    
    if len(gto_centers_2d) > 0:
        gto_centers_2d = gto_centers_2d.reshape(-1, 2)

    combined_alphas = np.concatenate([gto_alphas_2d, hybrids_alpha])
    combined_centers = np.concatenate([gto_centers_2d, hybrids_center])
    combined_labels = gto_labels_2d + hybrids_label
    
    C_z = np.zeros((n_gto, Nz))
    for i, g in enumerate(gtos):
        C_z[i, :] = evaluate_gto_z_on_grid(g, z_grid)
    
    eri_2d_blocks = {}
    for h in range(Nz):
        delta = h * dz
        eri_all = eri_2d_cartesian_with_p(combined_alphas, combined_centers, combined_labels, delta_z=delta)
        eri_block = eri_all[:n_gto, n_gto:, :n_gto, n_gto:]
        eri_2d_blocks[h] = eri_block
        
    return eri_2d_blocks, C_z