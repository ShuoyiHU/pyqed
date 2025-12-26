from pyqed.dvr.dvr_1d import SineDVR
from pyqed import ket2dm
from pyqed.qchem.dvr import RHF1D

import numpy as np
import scipy
from scipy.sparse.linalg import eigsh

#import proplot as plt
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax.scipy.special import erf, erfc
from jax.numpy import exp
from opt_einsum import contract

import pyqed
    
pi = jnp.pi


class Gaussians:
    def __init__(self, alpha=1, x=0):
        self.alpha = alpha
        self.center = x
        return

class Gaussian:
    """
    2D Real GWP
    """
    def __init__(self, alpha=1, center=0, ndim=3):

        if isinstance(alpha, (float, int)):
            alpha = np.array([alpha, ] * ndim)
        self.alpha = alpha

        if isinstance(center, (float, int)):
            center = np.array([center, ] * ndim)
        self.center = center

        self.ndim = ndim

class STO:
    """
    Contracted Gaussians for Slater-type orbitals
    """
    def __init__(self, n, c=None, g=None):
        self.n = n      # the number of GTOs
        self.d = self.c = np.array(c)      # contraction coefficents
        self.g = g      # primitive GTOs
        return

class ContractedGaussian:
    """
    Contracted Gaussian basis set for Slater-type orbitals
    """
    def __init__(self, n, c=None, g=None):
        self.n = n      # the number of GTOs
        self.d = self.c = np.array(c)      # contraction coefficents
        self.g = g      # primitive GTOs
        return


def sto_3g(center, zeta):

    scaling = zeta ** 2

    return ContractedGaussian(3, [0.444635, 0.535328, 0.154329],
               [Gaussian(scaling*0.109818, center),
                Gaussian(scaling*0.405771, center),
                Gaussian(scaling*2.22766, center)])

def sto_6g(center, zeta=1):

    c = [0.9163596281E-02, 0.4936149294E-01,  0.1685383049E+00, 0.3705627997E+00,\
         0.4164915298E+00, 0.1303340841E+00]

    a = [0.3552322122E+02, 0.6513143725E+01, 0.1822142904E+01, 0.6259552659E+00, \
      0.2430767471E+00, 0.1001124280E+00]

    g = [Gaussian(alpha=a[i], center=center) for i in range(6)]

    return ContractedGaussian(6, c, g)

def overlap_1d(aj, qj, ak, qk):
    """
    overlap between two 1D Gaussian wave packet

    .. math::

        g(x) = (2 \alpha/pi)^{1/4} * exp(-\alpha (x-x_0)^2)

    """
    # aj = g1.alpha
    # ak = g2.alpha
    # x = g1.center
    # y = g2.center

    # aj, x = g1
    # ak, y = g2


    dq = qk - qj

    result = (aj*ak)**0.25 * jnp.sqrt(2./(aj+ak)) * jnp.exp(    \
            -aj*ak/(aj+ak) * (dq**2) )
    return result

def overlap_2d(gj, gk):
    """
    overlap between two GWPs defined by {a,x,p}
    """

    aj, qj = gj.alpha, gj.center
    ak, qk = gk.alpha, gk.center

    tmp = 1.0
    for d in range(2):
        tmp *= overlap_1d(aj[d], qj[d], ak[d], qk[d])
        # tmp *= overlap_1d(gj,gk)

    return tmp

def sliced_contracted_gaussian(basis, z, ret_s=False):
    """


    Parameters
    ----------
    basis : TYPE
        contracted sliced Gaussian basis set.
    z : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    basis : TYPE
         contracted sliced Gaussian basis set

    """

    # scaling = zeta ** 2

    # sto = STO(3, [0.444635, 0.535328, 0.154329],
    #            [Gaussian(scaling*0.109818, center),
    #             Gaussian(scaling*0.405771, center),
    #             Gaussian(scaling*2.22766, center)])

    # absorb the exp(-a *z**2) part into coefficients
    n = basis.n
    sliced_basis = STO(basis.n)

    if isinstance(z, float):
        z = [z]
    nz = len(z)

    g = []
    c = np.zeros((n,nz))

    for i in range(basis.n):

        # g = basis.g[i]
        a = basis.g[i].alpha
        r0 = basis.g[i].center

        # print(a, r0)
        #
        c[i] = basis.c[i] * exp(-a[2] * (z-r0[2])**2) * (2*a[2]/np.pi)**0.25

        # reduce the dimension to 2
        g.append(Gaussian(center = r0[:2], alpha = a[:2], ndim=2))



    # renormalize the sliced basis
    # sto.d *= normalize(z)

    # # overlap between 2D Gaussians

    nb = n
    s = np.eye(n)
    for i in range(nb):
        for j in range(i):
            s[i, j] = overlap_2d(g[i], g[j])
            s[j, i] = s[i, j]


    norm = np.einsum('ia, ij, ja -> a', np.conj(c), s, c)
    c = np.einsum('ia,a -> ia', c, np.sqrt(1./norm))


    sliced_basis = [ContractedGaussian(n, g=g, c=c[:,i]) for i in range(nz)]

    if ret_s:
        return sliced_basis, s
    else:
        return sliced_basis



def sliced_eigenstates(mol, basis, z, k=1, contract=True):
    """


    Parameters
    ----------
    basis : ContractedGaussian obj or list of Gaussians
        CGBF.

    z : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    basis : TYPE
        unnormalized contracted Gaussian basis set

    """

    # scaling = zeta ** 2

    # sto = STO(3, [0.444635, 0.535328, 0.154329],
    #            [Gaussian(scaling*0.109818, center),
    #             Gaussian(scaling*0.405771, center),
    #             Gaussian(scaling*2.22766, center)])

    assert isinstance(basis, ContractedGaussian)
    # if isinstance(basis, ContractedGaussian):
        # if the basis is a single CGBF, then we use the primitive Gaussians as
        # the basis set for the transversal SE

    # absorb the exp(-a *z**2) part into coefficients
    n = basis.n
    sliced_basis = ContractedGaussian(n)

    gs = []
    # c = np.zeros(n)
    for i in range(basis.n):

        # g = basis.g[i]
        a = basis.g[i].alpha
        r0 = basis.g[i].center

        # c[i] = basis.c[i] * exp(-a[2] * (z-r0[2])**2) * (2*a[2]/np.pi)**0.25

        # reduce the dimension to 2
        gs.append(Gaussian(center = r0[:2], alpha = a[:2], ndim=2))


    # build overlap matrix
    S = np.eye(n)
    for i in range(n):
        for j in range(i):
            S[i,j] = overlap_2d(gs[i], gs[j])
            S[j, i] = S[i, j]

    # build the kinetic energy
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            K[i, j] = kin_2d(gs[i], gs[j])
            if i != j: K[j, i] = K[i, j]

    # build the potential energy
    V = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):

            for a in range(mol.natom):
                V[i, j] += electron_nuclear_attraction(gs[i], gs[j], (z - mol.atom_coord(a)[2]))

            if i != j: V[j, i] = V[i, j]

    H = K + V
    E, U = eigsh(H, k=k, M=S, which='SA')

    # print(E)
    # renormalize the sliced basis
    # sto.d *= normalize(z)
    # for n in range(k):
    if contract:
        sliced_basis = [ContractedGaussian(n, U[:,m], gs) for m in range(k)]

        return E, U, sliced_basis
    else:
        return E, U

    # elif isinstance(basis, list):
    #     # a list of CGBFs, we use the CGBF as the basis without unfolding
    #     # to the primitive GB
    #    pass




#for zn, renormalize 2D sto-3g basis
# sto = sto_3g(center=(0,0,0), zeta=1.24)
# nb = sto.n
# basis = sliced_contracted_gaussian(sto, z=2)



def normalize(cg, s=None):
    # basis = sto_3g_hydrogen(0)
    # a = [basis.g[i].alpha for i in range(basis.n)]
    if s is None:
        nb = cg.n
        s = np.eye(nb)
        for i in range(nb):
            for j in range(i):
                s[i, j] = overlap_2d(cg.g[i], cg.g[j])
                s[j, i] = s[i, j]

    c = cg.c

    norm = np.conj(c) @ s @ c
    cg.c *= np.sqrt(1./norm)
    return cg



def sto_3g_hydrogen(center=(0, 0, 0)):

    return sto_3g(center, zeta=1.24)

# basis = sto_3g_hydrogen()






#for zn, renormalize 2D sto-3g basis

# def normalize(z):
#     basis = sto_3g_hydrogen(0)
#     a = [basis.g[i].alpha[2] for i in range(basis.n)]

#     sum = 0
#     for i in range(basis.n):
#         for j in range(basis.n):
#             sum += np.exp(-a[i]*z**2)*np.exp(-a[j]*z**2)*basis.d[i]*basis.d[j]*overlap_2d(basis.g[i],basis.g[j])
#     return np.sqrt(1./sum)
#print("normalize = ",normalize(0))
'''
def sto_3g_2d(z):

    scaling = 1.24 ** 2
    N = normalize(z)

    return STO(3, [0.444635/z, 0.535328/z, 0.154329/z],
               [Gaussians(scaling*0.109818, 0),
                Gaussians(scaling*0.405771, 0),
                Gaussians(scaling*2.22766, 0)])
'''
def overlap_sto(b1,b2, s=None):
    if s is None:
        sum = 0.
        # N1 = normalize(z1)
        # N2 = normalize(z2)
        # a = [basis.g[i].alpha for i in range(basis.n)]

        for i in range(b1.n):
            for j in range(b2.n):
                sum += b1.c[i]*b2.c[j]*overlap_2d(b1.g[i],b2.g[j])
        return sum
    else:
        return np.conj(b1.c) @ s @ b2.c



def kin_1d(aj, qj, ak, qk):
    """
    kinetic energy matrix elements between two 1D GWPs
    """
    # aj = g1.alpha
    # ak = g2.alpha
    # qj = g1.center
    # qk = g2.center
    d0 = aj*ak/(aj+ak)
    l = d0 * overlap_1d(aj, qj, ak, qk)
    return l

def kin_2d(gj, gk):
    """
    kinetic energy matrix elements between two multidimensional GWPs
    """

    aj, qj = gj.alpha, gj.center
    ak, qk = gk.alpha, gk.center

    ndim = 2

    # overlap for each dof
    S = [overlap_1d(aj[d], qj[d], ak[d], qk[d]) \
         for d in range(ndim)]


    K = [kin_1d(aj[d], qj[d], ak[d], qk[d])\
         for d in range(ndim)]
    # S = [overlap_1d(gj,gk) \
    #      for d in range(ndim)]


    # K = [kin_1d(aj[d], qj[d], ak[d], qk[d])\
    #      for d in range(ndim)]

    res = 0
    for d in range(ndim):
        where = [True] * ndim
        where[d] = False
        res += K[d] * np.prod(S, where=where)

    return res

'''
def electron_nuclear_attraction(g1, g2, z):
    #i, q, r0 = g
    q = g1.alpha + g2.alpha
    b = z**2
    x = b * q



    return - jnp.sqrt(q/np.pi) * jnp.exp(x) * erfc(np.sqrt(x))
'''
def electron_nuclear_attraction(g1, g2, z):
    #i, q, r0 = g
    #q = g1.alpha + g2.alpha
    #b = z**2
    #x = b * q

    aj = g1.alpha[0]
    ak = g2.alpha[0]
    # print('xxxx', np.exp((aj+ak)*(z**2)) )
    # print('xxx', erfc(np.sqrt((aj+ak)*z**2)))

    x = np.sqrt((aj+ak)*(z**2))

    return  -2 * np.sqrt(aj*ak * np.pi/(aj+ak)) * scaled_erfc(x)

        # p = 0.47047
        # a1 = 0.3480242
        # a2 = -0.0958798
        # a3 = 0.7478556
        # t = 1/(1 + p * x)

        # return  -2 * np.sqrt(aj*ak * np.pi/(aj+ak)) * (a1 * t + a2 * t**2 + a3 * t**3)

    #result = - jnp.sqrt(q/np.pi) * jnp.exp(x) * erfc(np.sqrt(x))
    # return result



def kin_sto(b1,b2):
    """

    Compute the kinetic energy oprator matrix elements between two STOs

    .. math::

        K_{ij} = \langle g_i | - \frac{1}{2} \nabla_x^2 + \nabla_y^2 |g_j\rangle

    Parameters
    ----------
    b1 : TYPE
        DESCRIPTION.
    b2 : TYPE
        DESCRIPTION.
    z1 : TYPE
        DESCRIPTION.
    z2 : TYPE
        DESCRIPTION.

    Returns
    -------
    sum : TYPE
        DESCRIPTION.

    """
    sum = 0.
    # N1 = normalize(z1)
    # N2 = normalize(z2)

    # a = [basis.g[i].alpha for i in range(basis.n)]

    for i in range(b1.n):
        for j in range(b2.n):
            sum += b1.c[i]*b2.c[j]*kin_2d(b1.g[i],b2.g[j])
    return sum

def nuclear_attraction_sto(b1, b2, z):
    """


    Parameters
    ----------
    b1 : TYPE
        DESCRIPTION.
    b2 : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.

    Returns
    -------
    V : TYPE
        DESCRIPTION.

    """
    V = 0.
    # N1 = normalize(z)

    # a = [0.16885615680000002, 0.6239134896, 3.4252500160000006]

    for i in range(b1.n):
        for j in range(b2.n):

            tmp = electron_nuclear_attraction(b1.g[i], b2.g[j], z)
            # print('xx', i, j , tmp)

            V += b1.c[i] * b2.c[j] * tmp
    return V




def electron_repulsion_integral(g1, g2, g3, g4, z): # g1, g2 -> z1; g3, g4 -> z2;


    alpha = g1.alpha[0]      #Gaussian wave package ~ (2*alpha/pi)*0.5 * exp(-alpha((x-x0)**2+(y-y0)**2))
    beta = g2.alpha[0]
    delta = g3.alpha[0]
    sigma = g4.alpha[0]

    # p = alpha + delta
    # q = beta + sigma
    # x = np.sqrt(p * q / (p + q) * z**2)
    # c = (2/pi)**2 * (alpha * beta * delta * sigma)**(1./4)   # normalize 2d GWPs

    # def two_electron_integral_gto(g1, g2, g3, g4, z): # g1, g2 -> z1; g3, g4 -> z2;


    p = alpha + beta
    q = delta + sigma
    x = np.sqrt(p * q / (p + q) * z**2)
    c = (2/pi)**2 * (alpha * beta * delta * sigma)**(1/2)   # normalize 2d GWPs

    return c * pi**2.5 / np.sqrt(p * q * (p + q)) * scaled_erfc(x)


def scaled_erfc(x):
    """
    ... math::

        e^{x^2} \text{erfc}(x)

    when x > cutoff, switch to an expansion

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    if abs(x) < 9:
        return jnp.exp(x**2) * erfc(x)
    else:
        p = 0.3275911
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        t = 1/(1 + p * x)

        return a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5


# class RHF1D:
#     """
#     restricited DVR-HF method in 1D
#     """
#     def __init__(self, mol, init_guess='hcore', dvr_type = 'sine', domain=None, nx=None): # nelec, spin):
#         # self.spin = spin
#         # self.nelec = nelec
#         self.mol = mol

#         self.T = None
#         self.hcore = None
#         self.fock = None

#         self.mol = mol
#         self.max_cycle = 100
#         self.tol = 1e-6
#         self.init_guess = init_guess

#         ###
#         self.x = None
#         self.nx = nx
#         self.domain = domain

#         self.mo_occ = None
#         self.mo_coeff = None
#         self.e_tot = None

#         self.e_nuc = mol.energy_nuc()

#         self.e_kin = None
#         self.e_ne = None
#         self.e_j = None
#         self.e_k = None
#         self.hcore = None

#         self.eri = None

#     # def create_grid(self, domain, level, endpoints=False):

#     #     x = discretize(*domain, level, endpoints=endpoints)

#     #     self.x = x
#     #     self.nx = len(x)

#     #     self.lx = domain[1]-domain[0]
#     #     # self.dx = self.lx / (self.nx - 1)

#     #     self.domain = domain

#     def get_eri(self):
#         """
#         electronc repulsion integral in DVR basis

#         Returns
#         -------
#         v : TYPE
#             DESCRIPTION.

#         """
#         nx = self.nx
#         x = self.x

#         v = soft_coulomb(0, self.mol.Re) * np.eye(nx)
#         for i in range(nx):
#             for j in range(i):
#                 d = np.linalg.norm(x[i] - x[j])
#                 v[i,j] = soft_coulomb(d, self.mol.Re)
#                 v[j,i] = v[i,j]

#         self.eri = v
#         return v

#     def get_veff(self, dm):
#         """
#         compute Hartree and Fock potential

#         Parameters
#         ----------
#         dm : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         """

#         return get_veff(self.eri, dm)

#     def get_hcore(self):
#         """
#         single point calculations

#         Parameters
#         ----------
#         R : float
#             proton position.

#         Raises
#         ------
#         ValueError
#             DESCRIPTION.

#         Returns
#         -------
#         w : TYPE
#             DESCRIPTION.
#         u : TYPE
#             DESCRIPTION.

#         """

#         # H(r; R)


#         nx = self.nx
#         # T
#         # origin method of calculate kinetic term

#         dvr = SineDVR(*self.domain, nx)

#         x = dvr.x
#         self.x = x


#         # tx = kinetic(self.x, dvr=self.dvr_type)
#         T = dvr.t()

#         self.T = T

#         # V_en
#         # Ra = self.left
#         # Rb = self.right
#         v = np.zeros((nx))
#         for i in range(nx):
#             r1 = np.array(x[i])
#             # Potential from all ions
#             v[i] = self.mol.v_en(r1)

#         # print("rhf v", v)

#         V = np.diag(v)

#         # v_sym = self.enforce_spin_symmetry(v)
#         # # print(v_sym.shape)
#         # V = np.diag(v_sym.ravel())

#         H = T + V
#         # H = self.imaginary_time_propagation(H)

#         if np.any(np.isnan(H)) or np.any(np.isinf(H)):
#             raise ValueError("H matrix contains NaNs or infs.")

#         return H

#     def energy_nuc(self):
#         return self.mol.energy_nuc()

#     def run(self):
#         # scf cycle
#         max_cycle = self.max_cycle
#         tol = self.tol

#         mol = self.mol

#         # Hcore (kinetic + v_en)
#         hcore = self.get_hcore()
#         self.hcore = hcore

#         # occ number
#         nocc = self.mol.nelectron // 2
#         mo_occ = np.zeros(self.nx)
#         mo_occ[:nocc] = 2

#         self.mo_occ = np.stack([mo_occ, mo_occ])
#         # print('mo_occ', self.mo_occ)

#         eri = self.get_eri()

#         if self.init_guess == 'hcore':

#             mo_energy, mo_coeff = eigh(hcore)
#             dm = make_rdm1(mo_coeff, mo_occ)


#             vhf = get_veff(eri, dm)
#             old_energy = energy_elec(dm, hcore, vhf)

#         print("\n {:4s} {:13s} de\n".format("iter", "total energy"))

#         nuclear_energy = mol.energy_nuc()

#         print('nuclear repulsion', nuclear_energy)

#         conv = False
#         for scf_iter in range(max_cycle):

#             # calculate the two electron part of the Fock matrix

#             vhf = self.get_veff(dm)
#             F = hcore + vhf


#             mo_energy, mo_coeff = eigh(F)
#             # print("epsilon: ", epsilon)
#             #print("C': ", Cprime)
#             # mo_coeff = C
#             # print("C: ", C)


#             # new density matrix in original basis
#             # P = np.zeros(Hcore.shape)
#             # for mu in range(len(phi)):
#             #     for v in range(len(phi)):
#             #         P[mu,v] = 2. * C[mu,0] * C[v,0]
#             dm = make_rdm1(mo_coeff, mo_occ)

#             electronic_energy = energy_elec(dm, hcore, vhf)



#             print("E_elec = ", electronic_energy)

#             total_energy = electronic_energy + nuclear_energy

#             logging.info("{:3} {:12.8f} {:12.4e} ".format(scf_iter, total_energy,\
#                    total_energy - old_energy))

#             if scf_iter > 2 and abs(old_energy - total_energy) < tol:
#                 conv = True
#                 print('SCF Converged.')
#                 break

#             old_energy = total_energy


#             #println("F: ", F)
#             #Fprime = X' * F * X
#             # Fprime = dagger(X).dot(F).dot(X)
#             #println("F': $Fprime")

#         self.mo_coeff = mo_coeff
#         self.mo_energy = mo_energy


#         if not conv: sys.exit('SCF not converged.')

#         self.e_tot = total_energy
#         print('HF energy = ', total_energy)

#         return total_energy

#     # def energy_elec(dm, h1e=None, vhf=None):
#     #     r'''
#     #     Electronic part of Hartree-Fock energy, for given core hamiltonian and
#     #     HF potential

#     #     ... math::
#     #         E = \sum_{ij}h_{ij} \gamma_{ji}
#     #           + \frac{1}{2}\sum_{ijkl} \gamma_{ji}\gamma_{lk} \langle ik||jl\rangle

#     #     Note this function has side effects which cause mf.scf_summary updated.
#     #     Args:
#     #         mf : an instance of SCF class
#     #     Kwargs:
#     #         dm : 2D ndarray
#     #             one-partical density matrix
#     #         h1e : 2D ndarray
#     #             Core hamiltonian
#     #         vhf : 2D ndarray
#     #             HF potential
#     #     Returns:
#     #         Hartree-Fock electronic energy and the Coulomb energy
#     #     Examples:
#     #     >>> from pyscf import gto, scf
#     #     >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
#     #     >>> mf = scf.RHF(mol)
#     #     >>> mf.scf()
#     #     >>> dm = mf.make_rdm1()
#     #     >>> scf.hf.energy_elec(mf, dm)
#     #     (-1.5176090667746334, 0.60917167853723675)
#     #     >>> mf.energy_elec(dm)
#     #     (-1.5176090667746334, 0.60917167853723675)
#     #     '''
#     #     # if dm is None: dm = mf.make_rdm1()
#     #     # if h1e is None: h1e = mf.get_hcore()
#     #     # if vhf is None: vhf = mf.get_veff(mf.mol, dm)
#     #     e1 = np.einsum('ij,ji->', h1e, dm).real
#     #     e_coul = np.einsum('ij,ji->', vhf, dm).real * .5
#     #     # mf.scf_summary['e1'] = e1
#     #     # mf.scf_summary['e2'] = e_coul
#     #     # logger.debug(mf, 'E1 = %s  E_coul = %s', e1, e_coul)
#     #     return e1+e_coul #, e_coul


#     def make_rdm1(mo_coeff, mo_occ, **kwargs):
#         '''One-particle density matrix in AO representation
#         Args:
#             mo_coeff : 2D ndarray
#                 Orbital coefficients. Each column is one orbital.
#             mo_occ : 1D ndarray
#                 Occupancy
#         Returns:
#             One-particle density matrix, 2D ndarray
#         '''
#         mocc = mo_coeff[:,mo_occ>0]
#     # DO NOT make tag_array for dm1 here because this DM array may be modified and
#     # passed to functions like get_jk, get_vxc.  These functions may take the tags
#     # (mo_coeff, mo_occ) to compute the potential if tags were found in the DM
#     # array and modifications to DM array may be ignored.
#         return np.dot(mocc*mo_occ[mo_occ>0], mocc.conj().T)
# --- add this helper near your SCF code ---
class PulayDIIS:
    def __init__(self, max_vec=8, start=2, eps=1e-14):
        self.max_vec, self.start, self.eps = max_vec, start, eps
        self.err, self.fock = [], []

    def update(self, F, D, S):
        # Commutator error in AO metric: e = F D S - S D F  (Hermitian if close to SCF)
        e = F @ D @ S - S @ D @ F
        self.err.append(e.reshape(-1))     # store as vector
        self.fock.append(F.copy())
        if len(self.err) > self.max_vec:
            self.err.pop(0); self.fock.pop(0)

    def extrapolate(self):
        m = len(self.err)
        if m < self.start:
            return self.fock[-1]
        # Build B matrix
        B = np.empty((m+1,m+1), dtype=float)
        for i in range(m):
            for j in range(m):
                B[i,j] = float(np.vdot(self.err[i], self.err[j]).real)
        B[:m,-1] = -1.0; B[-1,:m] = -1.0; B[-1,-1] = 0.0
        rhs = np.zeros(m+1); rhs[-1] = -1.0
        # Regularize to avoid singular B
        try:
            coeff = np.linalg.solve(B + self.eps*np.eye(m+1), rhs)[:m]
        except np.linalg.LinAlgError:
            return self.fock[-1]  # fallback
        Fmix = sum(c*F for c, F in zip(coeff, self.fock))
        # Symmetrize to kill tiny numerical skew
        return 0.5*(Fmix + Fmix.T)

def _project_occ(Cprime, nocc):
    Cocc = Cprime[:, :nocc]
    return Cocc @ Cocc.T  # in orthonormal basis


class RHF_CG(RHF1D):
    def __init__(self, mol, nz, zrange, norb=1, basis='sto-6g', sliced_basis='no'):
        """


        Parameters
        ----------
        mol : TYPE
            DESCRIPTION.
        nz : TYPE
            DESCRIPTION.
        zrange : TYPE
            DESCRIPTION.
        norbs : TYPE, optional
            transversal orbs for each slice. The default is 1.
        basis : TYPE, optional
            DESCRIPTION. The default is 'sto-6g'.
        sliced_basis : TYPE, optional
            DESCRIPTION. The default is 'no'.

        Returns
        -------
        None.

        """
        self.zrange = zrange
        self.nz = nz
        self.basis = basis
        self.sliced_basis = sliced_basis
        self.norb = norb
        self.mol = mol

        ###
        self.hcore = None
        self._eri = None

    @property
    def eri(self):
        return self._eri

    def build(self):

        # L = self.L
        nz = self.nz

        dvr_z = SineDVR(npts=nz, *self.zrange)

        z = dvr_z.x
        kz = dvr_z.t()


        # print("z = ",z)

        # T = np.zeros((nz, no, no))
        natom = self.mol.natom
        R = self.mol.atom_coords()

        # 3D atom-centered STOs
        # sto = []
        # for n in range(natom):
        #     sto.append(sto_6g(R[n]))
        sto = sto_6g((0,0,0))


        # basis = [sliced_contracted_gaussian(sto, z[i]) for i in range(nz)]

        basis, s = sliced_contracted_gaussian(sto, z, ret_s=True)

        # # overlap between 2D Gaussians
        # b = basis[0]
        # nb = b.n
        # s = np.eye(nb)

        # for i in range(nb):
        #     for j in range(i):
        #         s[i, j] = overlap_2d(b.g[i], b.g[j])
        #         s[j, i] = s[i, j]

        # for b in basis:

        #     C = normalize(b, s)
        #     # for g in basis.g:
        #     b.c = np.array(b.c) * C


        # sliced natural orbitals

        # if self.sliced_basis == 'no':
        #     for b in basis:
        #         rho = s @ ket2dm(b.c) @ s
        #         noon, no = eigsh(rho, M=s, k=self.norbs, which='LM')
        #         print(noon)

        #         b.c = no[:,0]

        # basis = nos

        # print(normalize(basis, s))


        # transversal kinetic energy matrix
        T = np.zeros(nz)
        for n in range(nz):

            b = basis[n]
            T[n] = kin_sto(b, b)

        T = np.diag(T)

        # attraction energy matrix

        v = np.zeros(nz)
        # b1 = sto_3g_hydrogen(0)
        # b2 = sto_3g_hydrogen(0)

        for i in range(nz):
            b = basis[i]
            for A in range(natom):
                v[i] += nuclear_attraction_sto(b, b, z[i] - R[A, 2])

        V = np.diag(v)


        # construct H'

        hcore = T + V
        #print("H_prime", H_prime)



        #overlap matrix

        S = np.eye(nz)
        for i in range(nz):
            for j in range(i):
                S[i, j] = overlap_sto(basis[i],basis[j], s)
                S[j, i] = S[i, j]


        Tz = np.einsum('ij, ij -> ij', kz, S) # Kz * S

        #print("Tz = ",Tz)
        H = Tz + hcore
        #print("H = ",H)

        self.hcore = H


    def run(self):
        if self.hcore is None:
            self.build()

        H = self.hcore

        if self.mol.nelec == 1:
            E, U = eigsh(H, k=1, which='SA')
            print("Ground state energy = ", E)
        else:
            pass

        return E, U


class Molecule(pyqed.qchem.Molecule):
    def __init__(self, atom, nz, zrange, norb=1, basis='sto-6g', sliced_basis='eigensates', **kwargs):
        """


        Parameters
        ----------
        mol : TYPE
            DESCRIPTION.
        nz : TYPE
            DESCRIPTION.
        zrange : TYPE
            DESCRIPTION.
        norbs : TYPE, optional
            transversal orbs for each slice. The default is 1.
        basis : TYPE, optional
            DESCRIPTION. The default is 'sto-3g'.
        sliced_basis : TYPE, optional
            DESCRIPTION. The default is 'no'.

        Returns
        -------
        None.

        """
        super().__init__(atom, **kwargs)

        self.zrange = zrange
        self.nz = nz
        self.basis = basis
        self.sliced_basis = sliced_basis
        self.norb = norb
        # self.mol = mol

        ###
        # self.hcore = None
        # self.eri = None


    def build(self):

        nz = self.nz

        dvr_z = SineDVR(npts=nz, *self.zrange)
        dz = dvr_z.dx

        # transversal orbs
        nstates = norb = self.norb

        dvr_z = SineDVR(npts=nz, xmin=-L, xmax=L)
        # dvr_z = SincDVR(20, nz)
        z = dvr_z.x
        Kz = dvr_z.t()

        # print("z = ",z)
        nstates = self.norb

        # T = np.zeros((nz, no, no))

        sto = sto_6g(0)
        nbas = sto.n
        # sto = []
        # for a in range(self.mol.natom):
        #     sto.append(sto_6g(self.mol.atom_coord(a)))


        # if self.sliced_basis == 'eigenstates':
            
        basis = []
        E = np.zeros((nz, nstates))
        C = np.zeros((nz, nbas, nstates))

        for i in range(nz):
            # for ao in sto:
            E[i, :], C[i], sliced_basis = sliced_eigenstates(self, sto, z[i], k=nstates)

            basis.append(sliced_basis)

            # print(E[i])

        # # overlap between 2D Gaussians
        b = basis[0][0]
        nb = b.n
        s = np.eye(nb)
        for i in range(nb):
            for j in range(i):
                s[i, j] = overlap_2d(b.g[i], b.g[j])
                s[j, i] = s[i, j]



        # basis, s = sliced_contracted_gaussian(sto, z, ret_s=True)

        # basis = [normalize(b) for b in basis]

            # C = normalize(b)
            # # for g in basis.g:
            # b.c = np.array(b.c) * C

        # print(normalize(basis, s))


        # # transversal kinetic energy matrix
        # T = np.zeros(nz)
        # for n in range(nz):

        #     b = basis[n]
        #     T[n] = kin_sto(b, b)

        # T = np.diag(T)

        # # attraction energy matrix

        # v = np.zeros(nz)
        # # b1 = sto_3g_hydrogen(0)
        # # b2 = sto_3g_hydrogen(0)

        # for i in range(nz):
        #     b = basis[i]
        #     v[i] = nuclear_attraction_sto(b, b, z[i])
        # V = np.diag(v)


        # # construct H'

        # hcore = T + V




        #overlap matrix

        S = np.eye(nz)
        # basis = [0]*nz
        # for i in range(nz):
        #     basis[i] = sto_3g_hydrogen(0)
        # construct S

        # for i in range(nz):
        #     for j in range(i):
        #         S[i, j] = overlap_sto(basis[i], basis[j], s)
        #         S[j, i] = S[i, j]
        # Tz = np.einsum('ij, ij -> ij', kz, S) # Kz * S


        # transversal overlap matrix
        S = np.zeros((nz, nz, nstates, nstates))

        for n in range(nz):
            S[n, n] = np.eye(nstates)

        for i in range(nz):
            for j in range(i):

                for u in range(nstates):
                    for v in range(nstates):
                        S[i, j, u, v] = overlap_sto(basis[i][u], basis[j][v], s=s)


                S[j, i] = S[i, j].conj().T


        #print("Tz = ",Tz)
        V = np.diag(E.flatten())

        size = nz * nstates
        self.nao = size 
        H = np.einsum('mn, mnba -> mbna', Kz, S).reshape(size, size) + V

        #print("H = ",H)
        self.hcore = H


        # build ERI between 2D GTOs
        # TODO: exploit the 8-fold symmetry
        eri_gto = np.zeros((nz, nbas, nbas, nbas, nbas))

        for n in range(nz):

            for i in range(nbas):
                g1 = b.g[i]
                for j in range(nbas):
                    g2 = b.g[j]
                    for k in range(nbas):
                        g3 = b.g[k]
                        for l in range(nbas):
                            g4 = b.g[l]

                            eri_gto[n, i,j,k,l] = electron_repulsion_integral(g1, g2, g3, g4, n * dz)

                            # if i != j: eri_gto[n, j, i, k,l] = eri_gto[n, i,j,k,l]
                            # if k != l: eri_gto[n, i,j, l, k] = eri_gto[n, i,j,k,l]

        # print('eri_gto', eri_gto[0])
        # print( C[0].T @ s @ C[0])
        # print(C[0, :, 1])
        
        # from GTOs to sliced orbs
        eri = np.zeros((nz, nz, norb, norb, norb, norb))

        for m in range(nz):
            for n in range(m, nz):

                eri[m, n] = contract('ijkl, ia, jb, kc, ld -> abcd', eri_gto[n-m], C[m].conj(), C[m], C[n].conj(), C[n])
                if m != n:
                    eri[n, m] = np.transpose(eri[m,n], (2,3,0,1))

        # print('eri', eri[0,0][1,1,1,1])
        self.eri = eri

        return self

    def run(self):

        if self.hcore is None:
            self.build()
        H = self.hcore

        E, U = eigsh(H, k=1, which='SA')
        print("Ground state energy = ", E)

        return E, U
    

def symmetrize_triangular_array(ut):
    return np.where(ut,ut,ut.T)





# from pyqed import Molecule
from pyqed.qchem.dmrg import SpinHalfFermionChain, DMRG
# ===============================
# Minimal RHF-SCF using Molecule.build() outputs
#  - Uses: mol.hcore (or mol.core), mol.eri, and mol.S or mol.S_blocks if present
#  - ERI convention assumed: eri[m,n,a,b,c,d] = (ma,mb | nc,nd)
# ===============================
import numpy as np
import matplotlib.pyplot as plt
import scienceplots, matplotlib as mpl

def _to_blocks(mat, nz, norb):
    # (nao,nao) -> (m,n,a,b)
    return mat.reshape(nz, norb, nz, norb).transpose(0, 2, 1, 3)

def _from_blocks(blocks):
    # (m,n,a,b) -> (nao,nao)
    nz, nz2, norb, norb2 = blocks.shape
    assert nz == nz2 and norb == norb2
    return blocks.transpose(0, 2, 1, 3).reshape(nz*norb, nz*norb)

def _orthogonalizer(S, rtol=1e-10, atol=1e-12):
    # Canonical orthogonalization X = S^{-1/2} on kept subspace
    S = 0.5*(S + S.T)
    w, U = np.linalg.eigh(S)
    wmax = float(w.max())
    keep = w > max(atol, rtol*wmax)
    if not np.any(keep):
        raise np.linalg.LinAlgError("Overlap is singular—no directions kept.")
    return U[:, keep] @ np.diag(1.0/np.sqrt(w[keep]))

def _make_density_from_XC(X, Cprime, nelec):
    nocc = nelec // 2
    Cocc_ao = X @ Cprime[:, :nocc]
    return 2.0 * (Cocc_ao @ Cocc_ao.T.conj())
    
def K_from_gto_full(C, eri_gto, Dblk):
    nz, nbas, M = C.shape
    Kblk = np.zeros((nz,nz,M,M))
    for m in range(nz):
        for n in range(nz):
            Δ = n - m
            Kblk[m,n] = np.einsum('ijkl,ia,jc,kb,ld,cd->ab',
                                  eri_gto[Δ], C[m].conj(), C[m], C[n].conj(), C[n],
                                  Dblk[m,n], optimize=True)
    return Kblk

def _vhf_from_eri(eri, D_ao, nz, norb):
    """
    Build V_HF = J - 1/2 K (AO) from eri[m,n,a,b,c,d]=(ma,mb|nc,nd) and AO density D.
    - Coulomb J contributes only to blocks with identical left-slice indices (m==m).
      J[m,m,a,b] = sum_{p,c,d} (m a, m b | p c, p d) * D[p,p,c,d]
    - Exchange K is block-full:
      K[m,n,a,b] = sum_{c,d} (m a, n d | n b, m c) * D[n,m,c,d]
                  → einsum index: 'mnadbc,nmcd->mnab'
    """
    Dblk = _to_blocks(D_ao, nz, norb)             # (m,n,c,d)
    Ddiag = np.zeros((nz, norb, norb))
    for p in range(nz):
        Ddiag[p] = Dblk[p, p]

    # J only on diagonal blocks (m==m)
    Jblk = np.zeros((nz, nz, norb, norb))
    for m in range(nz):
        # eri[m] axes: (p,a,b,c,d)
        Jblk[m, m] = np.einsum('pabcd,pcd->ab', eri[m], Ddiag, optimize=True)

    # K on all (m,n): K[m,n,a,b] = sum_{c,d} eri[m,n,a,d,b,c] * D[n,m,c,d]
    Kblk = np.einsum('mnadbc,nmcd->mnab', eri, Dblk, optimize=True)
    Vblk = Jblk - 0.5 * Kblk
    return _from_blocks(Vblk), _from_blocks(Jblk), _from_blocks(Kblk)

def _total_e(H, Vhf, D, Enuc):
    E1 = float(np.einsum('ij,ji->', H, D))
    E2 = 0.5 * float(np.einsum('ij,ji->', Vhf, D))
    return E1 + E2 + Enuc
def scf_energy_from_mol(
    mol, tol=1e-8, max_cycle=100,
    damping=0.0,                # with DIIS, start at 0
    use_diis=True, diis_start=3, diis_max=8,
    level_shift=0.5,            # Eh, applied to virtuals in orthonormal basis
    kT=0.0,                     # finite-T smearing (Eh). Keep 0 unless needed.
    verbose=True
):
    H = getattr(mol, 'hcore', getattr(mol, 'core', None))
    if H is None: raise ValueError("Molecule has no hcore/core. Call mol.build() first.")
    eri = getattr(mol, 'eri', None)
    if eri is None: raise ValueError("Molecule has no eri. Call mol.build() first.")

    nz, norb = mol.nz, mol.norb
    nao = H.shape[0]

    # Overlap
    if hasattr(mol, 'S') and mol.S is not None:
        S = 0.5*(mol.S + mol.S.T)
    elif hasattr(mol, 'S_blocks') and mol.S_blocks is not None:
        S = _from_blocks(mol.S_blocks)
        S = 0.5*(S + S.T)
    else:
        S = np.eye(nao)

    # Electrons (closed shell)
    try:
        nelec = int(getattr(mol, 'nelectron'))
    except Exception:
        charges = np.asarray(mol.atom_charges(), float)
        charge = getattr(mol, 'charge', 0) or 0
        nelec = int(round(charges.sum() - charge))
    if nelec % 2 != 0:
        raise ValueError(f"RHF requires even electron count; got nelec={nelec}")
    nocc = nelec // 2
    Enuc = mol.energy_nuc()

    # Orthonormalize AO space
    X = _orthogonalizer(S)
    Hp = X.T @ (0.5*(H+H.T)) @ X
    eps, Cprime = np.linalg.eigh(Hp)

    def _occupations(eigs):
        if kT <= 0.0:
            occ = np.zeros_like(eigs); occ[:nocc] = 2.0
            return occ
        # finite-T (Fermi-Dirac) smearing, total occupancy = 2*nocc
        # Solve mu by bisection
        target = 2.0*nocc
        lo, hi = eigs.min()-50*kT, eigs.max()+50*kT
        for _ in range(60):
            mu = 0.5*(lo+hi)
            f = 2.0/(1.0 + np.exp((eigs-mu)/kT))
            if f.sum() > target: lo = mu
            else: hi = mu
        return 2.0/(1.0 + np.exp((eigs-mu)/kT))

    occ = _occupations(eps)
    D = _make_density_from_XC(X, Cprime, nelec) if kT<=0 else (X @ (Cprime*occ) @ Cprime.T @ X.T)

    diis = PulayDIIS(max_vec=diis_max, start=diis_start) if use_diis else None

    Eprev = None
    Ehist = []
    Cprime_prev = Cprime.copy()  # for level shift projector
    two_cycle_seen = False

    for it in range(1, max_cycle+1):
        # Build V_HF = J - 1/2 K  (AO)
        Vhf, Jao, Kao = _vhf_from_eri(eri, D, nz, norb)
        F = 0.5*((H + Vhf) + (H + Vhf).T)  # enforce Hermiticity

        # DIIS on Fock (AO)
        if diis is not None:
            diis.update(F, D, S)
            F = diis.extrapolate()

        # Transform to orthonormal basis
        Fp = X.T @ F @ X

        # Virtual level shift (use previous occupied subspace)
        if level_shift and it >= 2:
            Pocc_prev = _project_occ(Cprime_prev, nocc)  # in orthonormal basis
            Pvirt_prev = np.eye(Pocc_prev.shape[0]) - Pocc_prev
            Fp = Fp + level_shift * Pvirt_prev

        # Diagonalize
        eps, Cprime = np.linalg.eigh(Fp)
        occ = _occupations(eps)
        if kT <= 0.0:
            D_new = _make_density_from_XC(X, Cprime, nelec)
        else:
            D_new = X @ (Cprime*occ) @ Cprime.T @ X.T

        # Simple density damping (usually set to 0 with DIIS)
        if damping != 0.0:
            D = (1.0 - damping) * D_new + damping * D
        else:
            D = D_new

        # Total energy
        Etot = _total_e(H, Vhf, D, Enuc)
        Ehist.append(Etot)
        if verbose:
            dE = np.inf if Eprev is None else Etot - Eprev
            print(f"SCF {it:3d}: E = {Etot:.12f}  dE = {dE:+.3e}")

        # Two-cycle detection & guard
        if it >= 3:
            if abs(Ehist[-1] - Ehist[-3]) < 1e-8 and abs(Ehist[-1] - Ehist[-2]) > 1e-5:
                # detected 2-cycle → harden stabilization once
                if not two_cycle_seen:
                    two_cycle_seen = True
                    # strengthen: raise level_shift & temporary damping
                    level_shift = max(level_shift, 0.8)
                    damping = max(damping, 0.3)
                    # optionally enable a tiny smearing for 1–2 steps
                    if kT == 0.0:
                        kT = 1e-3  # Eh ~ 0.027 eV
                    if verbose:
                        print("  (two-cycle detected → boosting level_shift, damping, temporary kT)")
        # Convergence
        if Eprev is not None and abs(Etot - Eprev) < tol:
            if verbose: print("SCF converged.")
            break
        Eprev = Etot
        Cprime_prev = Cprime

    return Etot, Ehist

# ===============================
# Sweep & plot (uses your Molecule class/build)
# ===============================
plt.style.use(['ieee'])
mpl.rcParams.update({
    "mathtext.fontset": "dejavusans",
    "font.family": "dejavusans",
})

N = 10
ds = np.linspace(0.4, 2.7, N)
d = ds[5]
d = 0.91

L = 64

# 100 H atoms from z=-49 to z=49 (inclusive), evenly spaced
zs = np.linspace(-49,49,20)
# zs = np.linspace(-49,49, 50)
atom = '; '.join(f'H 0 0 {z:.6g}' for z in zs) + '; '

# nz_list = [32, 64, 128,256]
nz_list = [128,256]
# nz_list = [128,256,512]
norb_list = [1, 2, 3, 4, 5, 6]
E_list = [[] for _ in nz_list]

for i, Nz in enumerate(nz_list):
    print(f"\n==== Nz = {Nz} ====")
    for M in norb_list:
        mol = Molecule(atom=atom, unit='b', nz=Nz, zrange=[-L, L], norb=M)
        mol.build()  # must set mol.hcore/core, mol.eri, and (ideally) mol.S or S_blocks
        # E_scf = mol.run()[0]
        E_scf, _ = scf_energy_from_mol(mol, tol=1e-8, max_cycle=100, damping=0.2, verbose=True)
        E_list[i].append(E_scf)

# Plot
for i, Nz in enumerate(nz_list):
    plt.plot(norb_list, E_list[i], '-o', label=f'Nz={Nz}')
plt.axhline(-6.89658918386865,  linestyle='--', label='STO-6G reference', color='r')
plt.axhline(-8.38785786892562,  linestyle='--', label='631g** reference', color='b')
plt.axhline(-8.48067467567427,  linestyle='--', label='cc-pVDZ reference', color='g')
plt.legend()
plt.xlabel("M (number of contracted Gaussians kept per z slice)")
plt.ylabel("RHF total energy (Eh)")
plt.title(r"$H_{20}$ chain , $L_z$=(-64,64) RHF SCF energy vs M")
# plt.title(r"$H_{20}$ chain , $L_z$=(-30,30) RHF SCF energy vs M")
plt.grid(True)
plt.tight_layout()
plt.savefig(r"H20_-64_64 chain RHF SCF energy vs M.png", dpi=2400, bbox_inches='tight', pad_inches=0.05)
# plt.savefig(r"H20 chain RHF SCF energy vs M.png", dpi=2400, bbox_inches='tight', pad_inches=0.05)
plt.show()



# # atom= """
# #     H 0, 0, -1.5; 
# #     H 0, 0, -0.2; 
# #     H 0, 0, 0.2;
# #     H 0, 0, 1.5; 
# #         """

# N = 10 # number of geometries
# ds = np.linspace(0.4, 2.7, N)

# d = ds[5]
# d = 0.91

# # e_hf = np.zeros(N)
# # e_fci = np.zeros(N)

# Hi = [0, 0, -3.6]
# Hf = [0, 0, 3.6]

# H = [0, 0, ]

# atom= 'H 0, 0, -3.6; \
#     H 0, 0, -{}; \
#     H 0, 0, {}; \
#     H 0, 0, 3.6'.format(d, d)
# import scienceplots
# import matplotlib as mpl
# # E(FCI) = -1.145929244977
# L = 7
# nz = 128
# norb = 3


# mol = Molecule(atom=atom, unit='b', nz=nz, zrange=[-L, L], norb=norb)
# print(mol.atom_coords())
# mol.build()
# print(mol.run()[0])

# # E, U = RHFSlicedEigenstates(nz, L, norbs=1).run()

# mf = RHF_CG(mol, nz=nz, zrange=[-L, L], norb=norb)




# # nz_list = [32, 64, 128]
# # norb_list = [1, 2, 3, 4, 5, 6]
# # E_list = [[] for _ in range(len(norb_list))]
# # for idx, Nz in enumerate(nz_list):
# #     for norb in norb_list:
# #         mol = Molecule(atom=atom, unit='b', nz=Nz, zrange=[-L, L], norb=norb)
# #         mol.build()
# #         E_list[idx].append(mol.run()[0])

# # plt.style.use(['ieee'])
# # plt.plot(norb_list, E_list[0], '-o', label='Nz=32')
# # plt.plot(norb_list, E_list[1], '-o', label='Nz=64')
# # plt.plot(norb_list, E_list[2], '-o', label='Nz=128')
# # plt.ylim(-1.8, -1.35)
# # # plt.axhline(-1.8988478912704, color='r', linestyle='--', label='STO-6G reference')
# # # # and -1.77574072985318 as ccpvqz reference for H4 sqrare
# # # plt.axhline(-2.02986485121015, color='g', linestyle='--', label='cc-pVDZ reference')
# # plt.legend()
# # plt.xlabel("M (number of contracted gaussian kept per z slice)")
# # plt.ylabel("RHF total energy (Eh)")
# # # Make math text use the same (serif) font as the rest of the figure.
# # mpl.rcParams.update({
# #     "mathtext.fontset": "dejavusans",   # alternatives: 'cm', 'dejavusans', 'stixsans'
# #     "font.family": "dejavusans",      # use 'sans-serif' if you prefer a sans look
# # })
# # plt.title(r"RHF energy for $H_4$ square a = 1.4 $a_0$, Lz=(-8,8)) vs M")
# # plt.grid()
# # #auto fit fig size
# # plt.tight_layout()
# # #save fig
# # plt.savefig("H4_lsqrare_RHF_energy_vs_M_multiple_Nz_bg.png", dpi=1200, bbox_inches='tight', pad_inches=0.04)
# # plt.show()








# # N = mol.nao
# # eri_new = np.zeros((nz, norb, nz, norb, nz, norb, nz, norb))

# # for n in range(nz):
# #     for m in range(nz):
# #         eri_new[n, :, n, :, m, :, m, :] = mol.eri[n, m]

# # eri_new = eri_new.reshape(N, N, N, N)



# # from renormalizer.mps import Mps, Mpo, gs
# # from renormalizer.utils import CompressConfig, CompressCriteria
# # from renormalizer.model import h_qc, Model

# # from pyqed.qchem.dvr.rhf import RHF1D
# # from pyqed.models.ShinMetiu2e1d import AtomicChain
# # from pyqed import au2angstrom
# # import logging

# # logger = logging.getLogger("renormalizer")
# # logger.setLevel(logging.INFO)
# # np.seterr(divide="warn")
# # np.set_printoptions(precision=10)

# # h1e = mol.hcore 
# # natom = mol.natom


# # sh, aseri = h_qc.int_to_h(h1e, eri_new)
# # basis, res_terms = h_qc.qc_model(sh, aseri, spatial_orb=True,
# #         sp_op=True, sz_op=True, one_rdm=True)
# # ham_terms = res_terms["h"]
# # sp_terms = res_terms["s_+"]
# # sz_terms = res_terms["s_z"]

# # model = Model(basis, ham_terms)
# # h_mpo = Mpo(model)
# # sp_mpo = Mpo(model, terms=sp_terms)
# # sz_mpo = Mpo(model, terms=sz_terms)
# # s2_mpo = sp_mpo @ sp_mpo.conj_trans() + sz_mpo @ sz_mpo - sz_mpo
# # logger.info(f"h_mpo: {h_mpo}")
# # logger.info(f"s2_mpo: {s2_mpo}")


# # nelec = [natom//2, natom//2]
# # # vconfig = CompressConfig(CompressCriteria.threshold,threshold=1e-6)

# # # D bond dimension
# # vconfig = 12

# # procedure = [[vconfig,0.5], [vconfig,0.3], [vconfig, 0.1]] + [[vconfig, 0]]*20
# # M_init = 50
# # nstates = 1


# # # state average
# # mps = Mps.random(model, nelec, M_init, percent=1.0, pos=True)
# # mps.optimize_config.procedure = procedure
# # mps.optimize_config.method = "2site"
# # mps.optimize_config.nroots = nstates
# # energies, mpss = gs.optimize_mps(mps, h_mpo)
# # energies  = energies[-1]

# # print(energies +mol.energy_nuc())


# # # mol.eri = symmetrize_triangular_array(mol.eri.reshape(nz, nz))

# # # dmrg = DMRG(mol, D=10)
# # # dmrg.run()

# # # h1e = mf.hcore.reshape(nz, nz, norb, norb)


# # # print(mf.eri[30, 30])
# # # # nslice = 32
# # # n = 15
# # # model = SpinHalfFermionChain(h1e[n,n], mf.eri[n,n])
# # # model.run(8)

# # # H  = model.jordan_wigner()
# # # print(H.shape)
# # # print(eigsh(H,k=6)[0])

# # # E, U  = rhf.run()
# # # print(E)

# # # sto = sto_3g_hydrogen()
# # # sliced_eigenstates(sto, z=0)

# # # g1 = Gaussians()
# # # g2 = Gaussians()


# # # z = np.linspace(-5,5)

# # # e = np.zeros(len(z))
# # # for n in range(len(z)):

# # #     e[n] = electron_nuclear_attraction(g1, g2, z[n])


# # # fig, ax = plt.subplots()
# # # ax.plot(z, e, '-o')







# # # for nz in range(2,n):
# # #     Energy.append(energy(L, nz))
# # #     z.append(nz)

# # # #print("z = ",z)
# # # print("Energy = ", Energy)

# # # plt.plot(z, Energy, 'b.-', alpha=0.5, linewidth = 1, label = 'slice basis, L = {}'.format(L))

# # # plt.legend()
# # # plt.xlabel('nz')
# # # plt.ylabel("Energy(a.u.)")
# # # plt.ylim(-1,0.25)
# # # plt.savefig("save/slice_1.png")

