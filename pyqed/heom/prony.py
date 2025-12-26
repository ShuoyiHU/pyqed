#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 11:00:40 2024


Prony decomposition of time-series data

@author: Zi-Hao Chen 


"""



import math
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt


def fit_J(w, res, expn, etal, sigma):
    for i in range(len(etal)):
        res += etal[i] / (expn[i] + sigma * 1.j * w)


def fit_t(t, expn, etal):
    res = 0
    for i in range(len(etal)):
        res += etal[i] * np.exp(-expn[i] * t)
    return res





# fft_ct = np.exp(-fft_t)

def prony_decomposition(x, fft_ct, nexp, scale=None):
    """
    decompose a function into a sum of exponentials 
    
    Refs
        Zi-Hao's thesis
        
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    fft_ct : TYPE
        DESCRIPTION.
    nexp : TYPE
        DESCRIPTION.

    Returns
    -------
    etal1 : TYPE
        DESCRIPTION.
    expn1 : TYPE
        DESCRIPTION.
    err : TYPE
        DESCRIPTION.
        

    """
    
    n = (len(x)-1)//2
    
    n_sample = n + 1 
    n_gamma_l2 = [nexp] # number of exponentials
    
    h = np.real(fft_ct)
    H = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        H[i, :] = h[i:n_sample + i]
    sing_vs, Q = LA.eigh(H)
    
    # del H
    phase_mat = np.diag(
        [np.exp(-1j * np.angle(sing_v) / 2.0) for sing_v in sing_vs])
    vs = np.array([np.abs(sing_v) for sing_v in sing_vs])
    Qp = np.dot(Q, phase_mat)
    sort_array = np.argsort(vs)[::-1]
    vs = vs[sort_array]
    Qp = (Qp[:, sort_array])
    
    nroots = 21
    # vs = vs[:20]
    # Qp = Qp[:, :20]
    
    vs = vs[:nroots]
    Qp = Qp[:, :nroots]
    
    for n_gamma in n_gamma_l2:
        
        print("len of gamma", n_gamma)
        
        gamma = np.roots(Qp[:, n_gamma][::-1])
        # gamma = np.roots(Qp[:, n_gamma][::-1])
        gamma_new = gamma[np.argsort(np.abs(gamma))[:n_gamma]]
        t_real = 2 * n * np.log(gamma_new)
        gamma_m = np.zeros((n_sample * 2 - 1, n_gamma), dtype=complex)
        for i in range(n_gamma):
            for j in range(n_sample * 2 - 1):
                gamma_m[j, i] = gamma_new[i]**j
        omega_real = np.dot(LA.inv(np.dot(np.transpose(gamma_m), gamma_m)),
                            np.dot(np.transpose(gamma_m), np.transpose(h)))
    
        # res_t = np.zeros(len(t), dtype=complex)
        # fit_t(fft_t, res_t, -t_real / scale, omega_real)
        # plt.plot(fft_t, np.real(fft_ct) - res_t)
        # plt.savefig("real_{}.pdf".format(n_gamma))
        # plt.clf()
    

    etal1 = omega_real
    expn1 = -t_real / scale
    
    # fft_t = x 
    plt.plot(x, np.real(fft_ct))
    
    plt.plot(x, fft_ct, label='Exact')
    # plt.plot(fft_t, fft_ct.imag)
    
    # res_t = np.zeros(len(fft_t), dtype=complex)
    res_t = fit_t(x, expn1, etal1)
    
    plt.plot(x, res_t, '--', label='Fit')
    # plt.plot(fft_t, res_t.imag, '--')
    
    # plt.xlim(0, 200)
    plt.legend()
    
    err = (np.abs(fft_ct - res_t) ** 2).sum()/len(x) # sum of squared residues
    
    return etal1, expn1, err




if __name__=='__main__':
    
    n = 10000 # 2N + 1 points 
    scale = 200000 # range [0, 80]
    
    # scale_fft = 1000
    # n_fft = 10000000
    
    # n_rate = (scale_fft * scale/ (4 * n)) # print data every n_rate points
    # print(n_rate)
    # n_rate = int(n_rate)
    
    # w = np.linspace(0, scale_fft * np.pi, n_fft + 1)[:-1]
    # dw = w[1] - w[0]
    # print(dw)
    
    # fft_t = 2 * np.pi * np.fft.fftfreq(len(w), dw)
    # fft_t = fft_t[(scale>=fft_t) & (fft_t >= 0)][::n_rate]
    
    # fft_ct = 1 / fft_t
    
    # fft_ct[0] = fft_ct[1]
    
    
    x = np.linspace(0, scale, 2*n+1)
    
    fft_ct = 1/np.sqrt(x+1)
    # fft_ct = 1/np.sqrt(x**2)
    
    # fft_ct[0] = fft_ct[1]
    
    etal1, expn1, err = prony_decomposition(x, fft_ct, 20,scale=scale)
    print(etal1, expn1)
    print(err)
    plt.show()



# ########### following is a bunch of testing that is not useful for prony fitting itself ###########


# import numpy as np
# from numpy.polynomial.hermite import hermgauss

# # ---------------- fast numeric integral (Gauss–Hermite) ----------------
# def integral_numeric_hermite(alpha, a, rho, N=500):
#     """
#     I(alpha,a,rho) = ∫_{-∞}^{∞} e^{-α z^2} / sqrt((z-a)^2 + ρ^2) dz
#     Evaluated via N-point Gauss–Hermite. O(N), very fast and accurate.
#     """
#     if alpha <= 0 or rho <= 0:
#         raise ValueError("Require alpha>0 and rho>0.")
#     x, w = hermgauss(N)                       # weight e^{-x^2}
#     s = x/np.sqrt(alpha) - a                  # z = x/sqrt(alpha)
#     vals = 1.0 / np.sqrt(s*s + rho*rho)
#     return (w @ vals) / np.sqrt(alpha)

# # ------------- analytic integral from Prony Gaussian sum ---------------
# import numpy as np
# from scipy.special import erfc

# # ---------- Analytic for LINEAR exponentials ----------
# # kernel ~ sum c_k e^{-gamma_k x} on x>=0  ->  1/sqrt((z-a)^2+rho^2) ≈ sum (c_k/rho) e^{-(gamma_k/rho)|z-a|}
# def integral_from_prony(alpha, a, rho, coeffs, gammas):
#     coeffs = np.asarray(coeffs, dtype=np.complex128)
#     gammas = np.asarray(gammas, dtype=np.complex128)

#     # Only use decaying terms (Re gamma > 0)
#     mask = np.real(gammas) > 0
#     coeffs = coeffs[mask]
#     gammas = gammas[mask]
#     if coeffs.size == 0:
#         return 0.0

#     beta = gammas / rho
#     rt = np.sqrt(alpha)
#     pref = np.sqrt(np.pi)/(2.0*rt)

#     # vectorized complex-safe evaluation, sum real part at the end
#     term1 = np.exp(-beta*a + (beta*beta)/(4.0*alpha)) * erfc(-rt*a + beta/(2.0*rt))
#     term2 = np.exp(+beta*a + (beta*beta)/(4.0*alpha)) * erfc(+rt*a + beta/(2.0*rt))
#     I_terms = (coeffs/rho) * pref * (term1 + term2)
#     return float(np.sum(I_terms).real)


# def exp_sum(x, eta, lam):
#     x   = np.asarray(x, float)
#     eta = np.asarray(eta, np.complex128)
#     lam = np.asarray(lam, np.complex128)
#     return (np.exp(-np.outer(x, lam)) @ eta).real
# def plot_potential_and_fit(eta, xi, x_end=200.0, n_plot=4000, x_train=None):
#     """
#     Plot only the original potential V(x)=1/sqrt(1+x^2) and
#     the fitted Gaussian sum, on [0, x_end]. No residuals.
#     """
#     x = np.linspace(0.0, float(x_end), int(n_plot))
#     V_exact = 1.0 / np.sqrt(1.0 + x*x)
#     V_fit   = exp_sum(x, eta, xi)

#     plt.figure()
#     plt.plot(x, V_exact, label="Exact")
#     plt.plot(x, V_fit,  "--", label="Fit")
#     if x_train is not None:
#         plt.axvline(float(np.max(x_train)), ls=":", lw=1, label="train end")
#     plt.xlabel("x"); plt.ylabel("Value")
#     plt.legend()
#     plt.tight_layout()
# import numpy as np
# from numpy.polynomial.hermite import hermgauss
# from scipy.special import erfc

# # ---------- numeric reference for 2e (1D reduced) ----------
# def two_electron_numeric(alpha, beta, a, b, rho, N=200):
#     """
#     I_num = sqrt(pi/P) * ∫ e^{-γ(u-Δ)^2}/sqrt(ρ^2+u^2) du.
#     Use Gauss–Hermite on t = sqrt(γ) (u-Δ).
#     """
#     P = alpha + beta
#     gamma = alpha*beta / P
#     if gamma <= 0 or rho <= 0:
#         raise ValueError("gamma>0 and rho>0 required.")
#     x, w = hermgauss(N)                    # ∫ e^{-x^2} ...
#     u = (x/np.sqrt(gamma)) + (a - b)       # u = Δ + x/√γ
#     vals = 1.0 / np.sqrt(rho*rho + u*u)
#     J = (w @ vals) / np.sqrt(gamma)
#     return np.sqrt(np.pi / P) * J

# # ---------- your working single-integral kernel (linear exp) ----------
# # kernel ~ sum c_k e^{-gamma_k x} on x>=0  ->  1/sqrt((z-a)^2+rho^2) ≈ sum (c_k/rho) e^{-(gamma_k/rho)|z-a|}
# def _single_from_prony_linear(alpha, a, rho, coeffs, gammas):
#     coeffs = np.asarray(coeffs, dtype=np.complex128)
#     gammas = np.asarray(gammas, dtype=np.complex128)
#     mask = np.real(gammas) > 0
#     coeffs = coeffs[mask]; gammas = gammas[mask]
#     if coeffs.size == 0:
#         return 0.0
#     beta = gammas / rho
#     rt = np.sqrt(alpha)
#     pref = np.sqrt(np.pi)/(2.0*rt)
#     term1 = np.exp(-beta*a + (beta*beta)/(4.0*alpha)) * erfc(-rt*a + beta/(2.0*rt))
#     term2 = np.exp(+beta*a + (beta*beta)/(4.0*alpha)) * erfc(+rt*a + beta/(2.0*rt))
#     I_terms = (coeffs/rho) * pref * (term1 + term2)
#     return float(np.sum(I_terms).real)

# # ---------- two-electron analytic via your kernel ----------
# def two_electron_from_prony(alpha, beta, a, b, rho, coeffs, gammas):
#     """
#     I_prony = sqrt(pi/P) * J,   J is the same closed-form as your single integral
#     with alpha→gamma and a→Δ, i.e. J = ∫ e^{-γ(u-Δ)^2} * approx-kernel du.
#     """
#     P = alpha + beta
#     gamma = alpha*beta / P         # width for the u-Gaussian
#     Delta = a - b
#     J = _single_from_prony_linear(gamma, Delta, rho, coeffs, gammas)
#     return np.sqrt(np.pi / P) * J

# # ---------- helper: decide which array is (coeffs) and which is (exponents) ----------
# def pick_linear_mapping(xgrid, target, A, B):
#     """Choose (coeffs,exps) so that target(x)≈Σ coeffs*exp(-exps*x) best on the given grid."""
#     gA = np.sum(A[None,:] * np.exp(-(B[None,:]) * xgrid[:,None]), axis=1)
#     gB = np.sum(B[None,:] * np.exp(-(A[None,:]) * xgrid[:,None]), axis=1)
#     def rmse(y, yref): 
#         y = np.asarray(y, float); yref = np.asarray(yref, float)
#         return float(np.sqrt(np.mean((y - yref)**2)))
#     rA, rB = rmse(gA.real, target), rmse(gB.real, target)
#     return (A, B) if (np.isnan(rB) or rA <= rB) else (B, A)

# # # ------------------------------- main ----------------------------------
# # if __name__=='__main__':
# #     # from prony import prony_decomposition

# #     # ---- your Prony fitting setup (unchanged) ----
# #     n = 1000  # 2N + 1 points
# #     scale = 80
# #     x = np.linspace(0, scale, 2*n+1)          # nonnegative grid
# #     fft_ct = 1/np.sqrt(x**2 + 1)              # target g(y)=1/sqrt(1+y^2)

# #     etal1, expn1, err = prony_decomposition(x, fft_ct, 17)
# #     print("Prony raw:\n  etal1 =", etal1, "\n  expn1 =", expn1)
# #     print("Prony error =", err)

# #     # ---- pick the correct interpretation (A or B) on the same grid ----
# #     eta_base, xi_base = etal1, expn1
# #     plot_potential_and_fit(etal1, expn1, x_end=200, n_plot=4000, x_train=x)
# #     plt.show()


# #     # ---- quick integral check vs fast numeric (Gauss–Hermite) ----
# #     alpha = 1.0
# #     cases = [
# #         (alpha, 0.0, 0.5),
# #         (alpha, 0.0, 1),
# #         (alpha, 0.0, 5),
# #         (alpha, 1.0, 0.5),
# #         (alpha, 1.0, 1),
# #         (alpha, 1.0, 5),
# #         (2, 1.0, 0.5),
# #         (2, 1.0, 1),
# #         (2, 1.0, 5),
# #     ]

# #     def rel_err(approx, ref):
# #         return np.inf if ref == 0 else abs((approx - ref)/ref)

# #     print("\n alpha     a       rho      I_num                 I_prony            abs.err      rel.err")
# #     for (al, a, rho) in cases:
# #         I_num   = integral_numeric_hermite(al, a, rho, N=180)
# #         I_prony = integral_from_prony(al, a, rho, eta_base, xi_base)
# #         print(f"{al:6.3f}  {a:6.3f}  {rho:6.3f}   {I_num: .12e}   {I_prony: .12e}  {abs(I_prony-I_num):.3e}  {rel_err(I_prony,I_num):.3e}")
# if __name__ == "__main__":
    

#     # your Prony fit (unchanged)
#     n = 1000
#     scale = 80.0
#     x = np.linspace(0.0, scale, 2*n+1)          # x >= 0
#     g = 1.0 / np.sqrt(1.0 + x*x)
#     M = 17
#     etal, expn, err = prony_decomposition(x, g, M)
#     # print("Prony raw:\n  etal =", etal, "\n  expn =", expn)
#     print("Prony error =", err)

#     # make sure we interpret outputs as linear-exponential model
#     coeffs, gammas = pick_linear_mapping(x, g, etal, expn)

#     # filter to decaying terms
#     good = np.real(gammas) > 0
#     coeffs, gammas = coeffs[good], gammas[good]

#     # cases: (alpha, beta, a, b, rho)
#     cases = [
#         (1.0, 1.0, 0.0, 1, 1),
#         (1.0, 1.0, 0.0, 2, 1),
#         (1.0, 1.0, 0.0, 5, 1),
#         (1.0, 1.0, 0.0, 2, 2),
#         (1.0, 1.0, 0.0, 2, 5),
#         (1.0, 0.5, 0.0, 1, 2),
#         (1.0, 0.5, 0.0, 1, 2),
#     ]

#     def relerr(a, b): return np.inf if b == 0 else abs((a-b)/b)

#     print("\n   alpha   beta      a       b      rho      I_numeric              I_prony               abs.err        rel.err")
#     for (al, be, a0, b0, rho) in cases:
#         I_num   = two_electron_numeric(al, be, a0, b0, rho, N=220)
#         I_prony = two_electron_from_prony(al, be, a0, b0, rho, coeffs, gammas)
#         print(f"{al:8.3f}{be:7.3f}{a0:8.3f}{b0:8.3f}{rho:8.3f}   {I_num: .12e}   {I_prony: .12e}   {abs(I_prony-I_num):.3e}  {relerr(I_prony,I_num):.3e}")

