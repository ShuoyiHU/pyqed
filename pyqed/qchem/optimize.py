#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 23:03:15 2025

@author: bingg
"""

import numpy as np 
from opt_einsum import contract 

def minimize(f, X0, args=(), tau=1, taum=1e-15, tauM=1e15, eta=0.85, rho1=0.5, delta=0.2, epsilon=1e-5):
    """
    Implicit Steepest Descent Method for Optimization 
    with Orthogonality Constraints (Implicit–SD)
    
    Parameters
    ----------
    f : TYPE
        DESCRIPTION.
    X0 : TYPE
        DESCRIPTION.
    tau : TYPE, optional
        DESCRIPTION. The default is 1.
    eta : TYPE, optional
        DESCRIPTION. The default is 0.5.
    delta : TYPE, optional
        DESCRIPTION. The default is 0.5.
    epsilon : TYPE, optional
        DESCRIPTION. The default is 1e-5.

    Returns
    -------
    TYPE
        DESCRIPTION.
        
    References
    ----------
    
    Optimization Lett. 2022, 16:1773

    """

    n, p = X0.shape 
    
    Q0 = 1
    k = 0
    C = f(X0, *args)
    Id = np.identity(n)
    
    X = X0
    
    # taum = 1
    # tauM = 2
    
    Q = Q0
    G = gradient(X0, *args)
    # print('gradient', G)
    
    df = grad(X, G)
    
    while norm(df) > epsilon:
        
        A = G @ X.T - X @ G.T
        Y = project(np.linalg.inv(Id + tau * A) @ X)
        
        while f(Y, *args) > C + rho1 * tau * (-1/2 * norm(A)**2):
            tau = tau * delta 
            Y = project(np.linalg.inv(Id + tau * A) @ X)
        
        Xnew = Y 
        Qnew = eta * Q + 1 
        
        v = f(Xnew, *args)
        # print('energy = ', v)
        
        Cnew = (eta * Q * C + v)/Qnew
        Gnew = gradient(Xnew, *args)

        df_new = grad(Xnew, Gnew)
        
        tau = stepsize(k+1, Xnew-X, df_new-df)
        tau = max(min(tau, tauM), taum)
        
        k += 1
        
        # update 
        X = Xnew 
        Q = Qnew 
        C = Cnew 
        G = Gnew
        df = df_new 

    return X, f(X, *args)



def norm(A):
    """
    Frobenius norm of matrix
    .. math::

        ||A||_F = \sqrt{ A^\dagger A }
    """
    return np.sqrt(np.trace(A.T.conj() @ A))

def grad(X, G=None):
    """
    Riemmann gradient

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if G is None:
        G = gradient(X)
    return G - X @ G.T @ X

def project(V):
    """
    projection to Siefel manifold by orthonormalization 

    .. math::

        orth(V) = VQ\Lambda^{-1/2}

    where Q and \Lambda are eigenvectors and eigenvalues of :math:`V^T V`.

    Refs
    ----
    JCTC 2020, 12, 6207

    Parameters
    ----------
    V : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    w, Q = np.linalg.eigh(V.T @ V)
    return V @ Q @ np.diag(1/np.sqrt(w)) @ Q.T

def orth(V):
    """
    projection to Siefel manifold by orthonormalization 

    .. math::

        orth(V) = VQ\Lambda^{-1/2}

    where Q and \Lambda are eigenvectors and eigenvalues of :math:`V^T V`.

    Refs
    ----
    JCTC 2020, 12, 6207

    Parameters
    ----------
    V : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    w, Q = np.linalg.eigh(V.T @ V)
    return V @ Q @ np.diag(1/np.sqrt(w))

def stepsize(k, dU, dG):
    """
    Barzilai−Borwein stepsize for update U matrix

    Parameters
    ----------
    k : TYPE
        DESCRIPTION.
    U : TYPE
        DESCRIPTION.
    Uprev : TYPE
        DESCRIPTION.
    G : TYPE
        DESCRIPTION.
    Gprev : TYPE
        DESCRIPTION.

    Returns
    -------
    tau : TYPE
        DESCRIPTION.

    """
    if k % 2 == 0:
        # even
        # dU = U - Uprev
        # dG = G - Gprev
        tau = abs(inner(dU, dG))/inner(dG, dG)

    else:
        # odd
        # dU = U - Uprev
        # dG = G - Gprev
        tau = inner(dU, dU)/abs(inner(dU, dG))

    return tau

def inner(a, b):
    return np.trace(a.T @ b)

def gradient(U, h1e, h2e, dm1, dm2):
    g = h1e @ U @ dm1.T + h1e.T @ U @ dm1  # these two terms are probably the same
    g += 0.5 * (contract('pqrs, qb, rc, sd, abcd -> pa', h2e, U, U, U, dm2) + \
        contract('pqrs, pa, rc, sd, abcd -> qb', h2e, U, U, U, dm2) + \
        contract('pqrs, pa, qb, sd, abcd -> rc', h2e, U, U, U, dm2) + \
        contract('pqrs, pa, qb, rc, abcd -> sd', h2e, U, U, U, dm2) )
    return g


def energy(U, h1e, eri, dm1, dm2):
    """
    electronic energy 

    Parameters
    ----------
    U : ndarray of (n, p < n/2) 
        transformation matrix
    h1e : TYPE
        core Hamiltonian in canonical MO
    eri : TYPE
        DESCRIPTION.
    dm1 : TYPE
        DESCRIPTION.
    dm2 : TYPE
        DESCRIPTION.

    Returns
    -------
    e : TYPE
        DESCRIPTION.

    """
    
    e = contract('pq, pa, qb, ab ->', h1e, U, U, dm1)  
    e += 0.5 * (contract('pqrs, pa, qb, rc, sd, abcd ->', eri, U, U, U, U, dm2))
    return e 

def kernel(mf, U0, max_steps=50, tol=1e-6):
    """
    complete active space orbital optimization with orthonomality constraint

    .. math::
        U^\top U = I_N

        E = \sum_{p,q=1}^N t_{pq} U_{pp'} U_{q q'} \gamma_{p'q'} +
        1/2 v_{pqrs} \Gamma_{p'q'r's'} U_{pp'}U_{qq'}U_{rr'}U_{ss'}

    where U is a M x N (M > N) matrix.

    .. math::
        U_{k+1} = orth(U_k - \tau_k G_k)

    where G_k = \nabla P(U_k) is the gradient.

    Parameters
    ----------
    h1e : TYPE
        DESCRIPTION.
    h2e : TYPE
        DESCRIPTION.
    U0: ndarray
        initial guess of orbitals
    dm1 : TYPE
        DESCRIPTION.
    dm2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    k = 0
    U = U0 # initial guess for U0

    # first FCI calculation
    mc = CASCI(mf, ncas=2, nelecas=2)
    mc.run(U)
    e_old = mc.e_tot


    while k < max_steps:
        
        # update CI coeff
        mc.run(U)
        
        if abs(mc.e_tot - e_old) < tol:
            print("E(CASSCF) = {}".format(mc.e_tot))
            break

        dm1, dm2 = mc.make_rdm12(0)
        h1e = mc.hcore
        eri = mc.eri_so[0, 0] # for spin-restricted calculation

        # update the MOs by updating U
        U, E = minimize(energy, U, args=(h1e, eri, dm1, dm2))
        
        k += 1

    return mc

if __name__=='__main__':
    

    from pyqed import Molecule
    from pyqed.qchem.mcscf import CASCI
    
    mol = Molecule(atom='H 0 0 0; H 0 0 1.2', unit='b', basis='631g')
    mol.build()

    mf = mol.RHF().run()
    C = mf.mo_coeff
    
    ncas=2
    mc = CASCI(mf, ncas, nelecas=2)
    mc.run()
    
    dm1, dm2 = mc.make_rdm12(0)
    dm1 = dm1.T
    
    h1e = mf.get_hcore_mo()
    eri = mf.get_eri_mo()
    
    # h1e_ = mc.hcore
    # print(h1e_[0])
    print(h1e.shape)
    
    # eri = mc.eri_so[0, 0] # for spin-restricted calculation
    nmo = mol.nmo

    U0 = np.zeros((nmo, ncas))
    for i in range(ncas): 
        U0[i, i] = 1
    
    # print('E= ',energy(U0, h1e, eri, dm1, dm2))

    U, E = minimize(energy, U0, args=(h1e, eri, dm1, dm2))
    
    k = 0
    while k < 3:
        mo_coeff = C @ U 
        mc.run(mo_coeff=mo_coeff)
        
        # kernel(mf, U0)
        
        dm1, dm2 = mc.make_rdm12(0)
        dm1 = dm1.T
        
        U, E = minimize(energy, U0, args=(h1e, eri, dm1, dm2))
        print(E + mol.energy_nuc())
        
        k += 1

    