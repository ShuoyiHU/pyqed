#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 22:14:36 2025

1D Gaussian integrals

@author: bingg
"""
import numpy as np
from math import sqrt, pi
from scipy.special import erfc, wofz
# =========================== Prony fit for 1/sqrt(1+z^2) and helpers ===========================

PRONY_ETA = np.array([
    -2.05657322e-04+1.07265382e-05j, -2.09719037e-04-1.01741393e-05j,
     4.91867233e-03+7.13890837e-03j,  4.92726734e-03-7.12070214e-03j,
     4.38710247e-02+6.80026316e-02j,  4.38432677e-02-6.80169372e-02j,
    -5.31514652e-01-4.54793324e-06j,  2.69985708e-01-1.19027193e-07j,
     3.09576045e-01+2.66398318e-07j,  2.59997087e-01-1.19809086e-07j,
     1.95795192e-01+2.14924609e-08j,  1.41482601e-01-2.85576900e-10j,
     1.00534536e-01-2.40585211e-10j,  7.01952921e-02-1.27661839e-10j,
     4.69804764e-02+6.96743963e-11j,  2.81299022e-02-2.12571498e-11j,
     1.16928334e-02+3.18424728e-12j
], dtype=complex)
PRONY_XI  = np.array([
    9.34571014e+00+8.01224100e+00j, 9.34571014e+00-8.01224100e+00j,
    7.30374826e+00+4.05527387e+00j, 7.30374826e+00-4.05527387e+00j,
    5.40947417e+00-1.65660863e+00j, 5.40947417e+00+1.65660863e+00j,
    3.52550274e+00+4.29987068e-14j, 1.64509647e+00-1.80893789e-14j,
    1.11887340e+00+3.23858138e-13j, 7.54456874e-01-3.48973866e-12j,
    5.02290327e-01+7.59197541e-12j, 3.27235412e-01-8.28803998e-12j,
    2.05069055e-01+4.95002104e-12j, 1.19861858e-01-3.45633424e-12j,
    6.16151650e-02+5.94130590e-12j, 2.43131108e-02-5.40278844e-12j,
    4.53897441e-03+1.92264766e-12j
], dtype=complex)


# ----------------- helpers for single normalized GWP density -----------------

def _dens_at(alpha, q, z0):
    """|g(z0)|^2 for normalized 1D GWP; p,s drop out."""
    return sqrt(alpha/pi) * np.exp(-alpha * (z0 - q)**2)

def _moment_single_norm(alpha, q, a, beta):
    """
    M_single(beta) = ∫ |g(z)|^2 e^{-beta |z-a|} dz
    Stable closed form using Faddeeva w(z)=wofz(z):
      let D = a - q, r = sqrt(alpha)
      u_minus = -r*D + beta/(2r)
      u_plus  = +r*D + beta/(2r)
      M = 0.5 * exp(-alpha*D^2) * [ wofz(1j*u_minus) + wofz(1j*u_plus) ]
    """
    r = sqrt(alpha)
    D = a - q
    u_minus = -r*D + beta/(2.0*r)
    u_plus  = +r*D + beta/(2.0*r)
    return 0.5 * np.exp(-alpha*D*D) * (wofz(1j*u_minus) + wofz(1j*u_plus))

def _prod_prefactor(alpha_mu, q_mu, alpha_nu, q_nu):
    """
    χμ χν product with normalized primitives (p=s=0):
      χμ(z)χν(z) = P0 * exp[-A (z - zc)^2]
      A  = 0.5*(αμ+αν)
      zc = (αμ qμ + αν qν) / (αμ + αν)
      P0 = (αμ αν)^{1/4}/sqrt(pi) * exp( -0.5(αμ qμ^2 + αν qν^2) + A*zc^2 )
    """
    am, an = alpha_mu, alpha_nu
    qm, qn = q_mu, q_nu
    A  = 0.5*(am + an)
    zc = (am*qm + an*qn) / (am + an)
    const = -0.5*(am*qm*qm + an*qn*qn) + A*zc*zc
    P0 = (am*an)**0.25 / sqrt(pi) * np.exp(const)
    return P0, A, zc


def _moment_pair(alpha_mu, q_mu, alpha_nu, q_nu, a, beta):
    """
    J_{μν}(beta) = ∫ χμ(z) χν(z) e^{-beta |z-a|} dz   (normalized real primitives)
    Stable closed form:
      A  = 0.5*(αμ+αν); zc = (αμ qμ + αν qν)/(αμ+αν); rA = sqrt(A); D = zc - a
      P0 = (αμ αν)^{1/4}/sqrt(pi) * exp( -0.5(αμ qμ^2 + αν qν^2) + A zc^2 )
      pref = 0.5 * sqrt(pi/A)
      u1 = -rA*D + beta/(2 rA)
      u2 = +rA*D + beta/(2 rA)
      J = P0 * pref * exp(-A D^2) * [ wofz(1j*u1) + wofz(1j*u2) ]
    """
    am, an = alpha_mu, alpha_nu
    qm, qn = q_mu, q_nu
    A  = 0.5*(am + an)
    zc = (am*qm + an*qn) / (am + an)
    rA = sqrt(A)
    D  = zc - a
    const = -0.5*(am*qm*qm + an*qn*qn) + A*zc*zc
    P0 = (am*an)**0.25 / sqrt(pi) * np.exp(const)
    pref = 0.5 * sqrt(pi/A)
    u1 = -rA*D + beta/(2.0*rA)
    u2 = +rA*D + beta/(2.0*rA)
    return P0 * pref * np.exp(-A*D*D) * (wofz(1j*u1) + wofz(1j*u2))

# =========================== Matrix elements ===========================



def kinetic_energy(aj, qj, pj, sj, ak, qk, pk, sk, mass=1):

    """
    kinetic energy matrix elements between two multidimensional GWPs

    .. math::

        T_{jk} = \langle g_j | - \frac{1}{2m} \grad^2 | g_k \rangle
    Calculates the kinetic energy matrix element between two multidimensional Gaussian wave packets (GWPs).
    Parameters:
        aj (float or complex): Width parameter of the first GWP.
        qj (float or np.ndarray): Center position of the first GWP.
        pj (float or np.ndarray): Momentum of the first GWP.
        sj (float or complex): Normalization or phase factor of the first GWP.
        ak (float or complex): Width parameter of the second GWP.
        qk (float or np.ndarray): Center position of the second GWP.
        pk (float or np.ndarray): Momentum of the second GWP.
        sk (float or complex): Normalization or phase factor of the second GWP.
        mass (float, optional): Mass of the particle. Default is 1.
    Returns:
        complex: The kinetic energy matrix element ⟨g_j| -½/m ∇² |g_k⟩ between the two GWPs.

    """

    p0 = (aj*pk + ak*pj)/(aj+ak)
    d0 = 0.5/mass * ( (p0+1j*aj*ak/(aj+ak)*(qj-qk))**2 + aj*ak/(aj+ak) )

    l = d0 * overlap(aj, qj, pj, sj, ak, qk, pk, sk)

    return l

def overlap(aj, x, px, sj, ak, y, py, sk):
    """
    overlap between two 1D GWPs <g_j|g_k>

    .. math::

        g(x) = (\alpha/\pi)^{1/4} e^{- \alpha/2 (x-q)^2 + ip(x-q) + i \theta}
    """
    dp = py - px
    dq = y - x

    return (aj*ak)**0.25 * np.sqrt(2./(aj+ak)) * np.exp(    \
            -0.5 * aj*ak/(aj+ak) * (dp**2/aj/ak + dq**2  \
            + 2.0*1j* (px/aj + py/ak) *dq) ) * np.exp(1j * (sk-sj))


def nuclear_attraction(x, y, nucleus_xyz, Z,
                       aj=None, qj=None, pj=0.0, sj=0.0,
                       cgf=None,
                       eta=PRONY_ETA, xi=PRONY_XI,
                       eps=1e-8):
    """
    ⟨Φ_z | - Z / sqrt((z - Z_n)^2 + ρ^2) | Φ_z⟩   at one DVR (x,y), one nucleus.

    Prony linear-exponential fit:
      1/sqrt(1+u^2) ~ Σ η_j exp(-ξ_j |u|), with u = (z-a)/ρ,  a = Z_n,  ρ = sqrt((x-Xn)^2 + (y-Yn)^2)

    Two options:
      - Single GWP (provide aj, qj; p,s ignored by density)
      - Contracted CGF (provide cgf with normalized primitives, p=s=0)

    Returns a real float.
    """
    Xn, Yn, Zn = float(nucleus_xyz[0]), float(nucleus_xyz[1]), float(nucleus_xyz[2])
    rho = float(np.hypot(x - Xn, y - Yn))
    a   = Zn

    if eta is None or xi is None or len(eta) == 0:
        raise ValueError("Prony (eta, xi) must be provided (linear exponential in |u|).")

    # ---- nucleus exactly on the grid (rho ~ 0): delta-sequence limit ----
    if rho < eps:
        coeff = np.sum(2.0 * eta / xi)  # complex allowed
        if cgf is None:
            if aj is None or qj is None:
                raise ValueError("Provide (aj,qj,...) for single-GWP case.")
            dens = _dens_at(float(aj), float(qj), a)
        else:
            alphas = np.asarray(cgf.alpha, float)
            qs     = np.asarray(cgf.center, float)
            cs     = np.asarray(cgf.coeff, complex)
            chi_a  = (alphas/pi)**0.25 * np.exp(-0.5*alphas*(a - qs)**2)  # real primitives
            phi_a  = np.vdot(cs, chi_a)   # cs* dot chi
            dens   = float(np.abs(phi_a)**2)
        val = np.real(coeff) * dens
        return float(-Z * val)

    # ---- rho > 0: sum over Prony terms with beta = xi / rho ----

    # ρ > 0: sum over Prony terms
    total = 0.0 + 0.0j

    # SINGLE-GWP
    if cgf is None:
        alpha = float(aj); q = float(qj)
        for (et, rate) in zip(eta, xi):
            beta = rate / rho              # KEEP COMPLEX !!!!1
            total += (et / rho) * _moment_single_norm(alpha, q, a, beta)
        return float(-Z * np.real(total))

    # CONTRACTED
    else:
        alphas = np.asarray(cgf.alpha, float)
        qs     = np.asarray(cgf.center, float)
        cs     = np.asarray(cgf.coeff, complex)
        for (et, rate) in zip(eta, xi):
            beta = rate / rho              # <-- KEEP COMPLEX
            s_j = 0.0 + 0.0j
            for mu in range(len(alphas)):
                am, qm, cm = alphas[mu], qs[mu], cs[mu]
                for nu in range(len(alphas)):
                    an, qn, cn = alphas[nu], qs[nu], cs[nu]
                    s_j += cm * cn * _moment_pair(am, qm, an, qn, a, beta)
            total += (et / rho) * s_j
        return float(-Z * np.real(total))

def electron_repulsion():
    pass




from pyqed.dvr.dvr_2d import DVR2

nx=15
ny=15

dvr = DVR2([-6, 6], [-6, 6], nx, ny)

# kinetic energy matrix elements 
t = dvr.t()
t = np.asarray(t.toarray() if hasattr(t, "toarray") else t)
t = t.reshape(nx, ny, nx, ny)





class CGF:
    def __init__(self, coeff, alpha, center):
        """
        a contracted Gaussian functions for z in the HF/DVR method. 

        Parameters
        ----------
        coeff : TYPE
            DESCRIPTION.
        alpha : TYPE
            DESCRIPTION.
        center : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.coeff = coeff 
        self.alpha = alpha 
        self.center = center 
        
    
        




# print(t.shape)