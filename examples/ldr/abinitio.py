#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:57:08 2024

@author: xiaozhu
"""

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
from pyqed import discretize, interval
from pyqed.dvr import SineDVR
from pyqed.ldr.ldr import ResultLDR
from pyqed import interval,cartesian_product, discretize
from pyqed.dvr.dvr_1d import HermiteDVR, SineDVR
import scipy.sparse as sp
from opt_einsum import contract
from tqdm import tqdm
import string
from scipy.linalg import inv
from scipy.sparse import eye
import logging
from pyscf import gto, scf, ci
from tqdm import tqdm
import pickle
from functools import reduce
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import inv



def gen_enisum_string(D):
    alphabet = list(string.ascii_lowercase)
    if (D > 10):
        raise ValueError('Dimension D = {} cannot be larger than 10.'.format(D))

    ini = "".join(alphabet[:D]) + 'x' + "".join(alphabet[D:2*D])+'y'
    einsum_string = ini
    for n in range(D):
        einsum_string += ','
        einsum_string += alphabet[n] + alphabet[n+D]

    return (einsum_string + ' -> ' + ini)




class LDRN:
    """
    many-dimensional many-state nonadiabatic conical intersection dynamics in
    DVR + LDR + SPO


    The required input to run is APES and electronic overlap matrix.


    This is extremely expansive, the maximum dimension should be < 4.

    """
    def __init__(self, domains, levels, ndim=3, nstates=2, x0=None, mass=None, \
                 dvr_type='sine', ref_geom=None):

        assert(len(domains) == len(levels) == ndim)

        self.domains = domains
        self.levels = levels

        self.L = [domain[1] - domain[0] for domain in domains]

        # x = []
        w = []
        dvr = []
        if dvr_type in ['sinc', 'sine']:
            # uniform grid
            for d in range(ndim):
                l = levels[d]
                # x.append(discretize(*domains[d], levels[d], endpoints=True))
                _w = [1/(2**l+1), ] * (2**l+1)
                w.append(_w)

        elif dvr_type == 'gauss_hermite':

            assert x0 is not None

            for d in range(ndim):
                _dvr = HermiteDVR(x0[d], levels[d])
                # x.append(_dvr.x)
                w.append(_dvr.w)
                dvr.append(_dvr.copy())

        else:
            raise ValueError('DVR {} is not supported. Please use sinc.')


        self.x = ref_geom
        self.w = w # weights
        self.dvr = dvr
        self.dx = [interval(_x) for _x in ref_geom]
        self.nx = [len(_x) for _x in ref_geom]

        self.dvr_type = [dvr_type, ] * ndim

        if mass is None:
            mass = [1, ] * ndim
        self.mass = mass

        self.nstates = nstates
        self.ndim = ndim

        # all configurations in a vector
        self.points = np.fliplr(cartesian_product(ref_geom))
        self.ntot = len(self.points)

        ###
        self.H = None
        self.K = self.T = None
        # self._V = None

        self._v = None
        self.exp_K = None
        self.exp_V = None
        self.exp_T = None # KEO in LDR
        self.wf_overlap = self.A = None
        self.apes = None


    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, v):
        assert(v.shape == (*self.nx, self.nstates, self.nstates))

        if abs(np.min(v)) > 0.1:
            raise ValueError('The PES minimum is not 0. Shift the PES.')

        self._v = v


    def buildK(self, dt):
        """
        For the kinetic energy operator with Jacobi coordinates

            K = \frac{p_r^2}{2\mu} + \frac{1}{I(r)} p_\theta^2

        Since the two KEOs for each dof do not commute, it has to be factorized as

        e^{-i K \delta t} = e{-i K_1 \delta t} e^{- i K_2 \delta t}

        where $p_\theta = -i \pa_\theta$ is the momentum operator.


        Parameters
        ----------
        dt : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        self.exp_K = []
        self.K = []

        for d in range(self.ndim):

            dvr = SineDVR(*self.domains[d], self.nx[d], mass=self.mass[d])

            Tx = dvr.t() # np.array(expKx).shape=(nx, ny)
            expKx = dvr.expT(dt) # np.array(Tx).shape=(nx, ny)

            self.exp_K.append(expKx.copy())
            self.K.append(Tx.copy())

        print('self.exp_K.shape=', np.array(self.exp_K).shape, 'self.K.shape=', np.array(self.K).shape)
        return self.exp_K, self.K



    def buildV(self, dt):
        """
        Setup the propagators appearing in the split-operator method.



        Parameters
        ----------
        dt : TYPE
            DESCRIPTION.

        intertia: func
            moment of inertia, only used for Jocabi coordinates.

        Returns
        -------
        None.

        """

        dt2 = 0.5 * dt
        self.exp_V = np.exp(-1j * dt * self.apes)

        self.exp_V_half = np.exp(-1j * dt2 * self.apes)

        return


    def gen_enisum_string(self, D):
        """
        Generate einsum string for computing the short-time propagator

        ij...a, ij...a, kl...b, lk...b -> ija, klb

        Parameters
        ----------
        D : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        alphabet = list(string.ascii_lowercase)
        if (D > 10):
            raise ValueError('Dimension D = {} cannot be larger than 10.'.format(D))

        s1 = "".join(alphabet[:D]) + 'x'
        # s2 = "".join(alphabet[:D]) + 'x' + "".join(alphabet[D:2*D])+'y'
        s3 = "".join(alphabet[D:2*D])+'y'
        s2 = s1 + s3

        einsum_string = s1 + ',' + s2 + ',' + s3 + '->' + s2

        return einsum_string


    def short_time_propagator(self, dt):

        if self.apes is None:
            print('building the adibatic potential energy surfaces ...')
            self.build_apes()

        self.buildV(dt)

        print('building the kinetic energy propagator')
        self.buildK(dt)


        if self.A is None:
            logging.info('building the electronic overlap matrix')
            self.build_ovlp()


        einsum_string = gen_enisum_string(self.ndim)
        exp_T = contract(einsum_string, self.A, *self.exp_K)


        einsum_string = self.gen_enisum_string(self.ndim)
        U = contract(einsum_string, self.exp_V_half, exp_T, self.exp_V_half)
        return U


    def buildH(self, dt):

        print('building the potential energy propagator')
        self.buildV(dt)

        print('building the kinetic energy propagator')
        self.buildK(dt)

        if self.A is None:
            logging.info('building the electronic overlap matrix')
            self.build_ovlp()

        size = np.prod(self.nx) * self.nstates # size = nx * ny

        einsum_string = gen_enisum_string(self.ndim)
        H = np.diag(self.apes.flatten()) + \
            contract(einsum_string, self.A, *self.K).reshape(size, size) #einsum_string = abxcdy,ac,bd -> abxcdy, self.A.shape=(63, 63, 1, 63, 63, 1), self.K.shape= (2, 63, 63)

        self.H = H

        return H


    def Hpsi(self, psi, A, K, Va):

        Kx = K[0]
        Ky = K[1]

        Kx_Ky = sp.kron(Kx, np.eye(self.nx[1])) + sp.kron(np.eye(self.nx[0]), Ky)
        Kx_Ky = Kx_Ky.toarray()
        K_total = Kx_Ky.reshape(self.nx[0], self.nx[1],self.nx[0], self.nx[1])

        K_total = contract('abxcdy, abcd -> abxcdy', A, K_total)
        Kpsi = contract('abxcdy, cdy -> abx', K_total, psi)

        Vpsi = Va*psi # Va is adiabatic potential energy matrix
        hpsi = Kpsi + Vpsi

        return -1j * hpsi



    def run(self, psi0, dt, nt, nout=1, t0=0, method='spo'):

        assert(psi0.shape == (*self.nx, self.nstates))

        r = ResultLDR(dx=self.dx, x=self.x, dt=dt, psi0=psi0, Nt=nt, t0=t0, nout=nout)
        r.psilist = [psi0]

        if method == 'spo':
            # Split-operator method for linear coordinates

            if self.H is None:
                self.buildH(dt)

            einsum_string = gen_enisum_string(self.ndim)
            exp_T = contract(einsum_string, self.A, *self.exp_K) #einsum_string = abxcdy,ac,bd -> abxcdy, #sself.exp_K.shape=(2, 63, 63)

            alphabet = list(string.ascii_lowercase)
            D = self.ndim
            _string = "".join(alphabet[:D]) + 'x' + "".join(alphabet[D:2*D])+'y, ' + \
                "".join(alphabet[D:2*D])+'y -> ' + "".join(alphabet[:D]) + 'x'

            psi = psi0.copy()
            psi = self.exp_V_half * psi
            for k in tqdm(range(nt//nout)):
                for kk in range(nout):

                    psi = contract(_string, exp_T, psi)
                    psi = self.exp_V * psi

                r.psilist.append(psi.copy())
            psi = self.exp_V_half * psi


        elif method == 'rk4':

            print('building the kinetic energy propagator')
            exp_K, K = self.buildK(dt)

            t = t0
            psi = psi0.copy()

            for k in range(nt//nout):
                for l in range(nout):

                    t += dt
                    psi = rk4(psi, self.Hpsi, dt, self.A, K, self.apes)

                r.psilist.append(psi.copy())

        return r


    def initialize_wavepacket(self, ref_geom, idx0, idx1, a, A, target_state=1):
        """
        Initialize a Gaussian wavepacket psi0 centered at position x0.

        Parameters:
        - ref_geom: list of two 1D arrays [x_grid, y_grid]
        - nx, ny: grid sizes in x and y directions
        - nstates: number of electronic states
        - idx0, idx1: index of initial position (e.g. [0.0, 1.71875])
        - a: 2x2 matrix (numpy array), Gaussian width matrix
        - target_state: the index of the adiabatic state (default is 1)
        - A: global electronic overlap matrix

        Returns:
        - psi0: complex array of shape (nx, ny, nstates)
        """
        ndim = self.ndim
        nx, ny = len(ref_geom[0]), len(ref_geom[1])
        psi0 = np.zeros((nx, ny, self.nstates), dtype=complex)
        for i in range(nx):
            for j in range(ny):
                x = np.array([ref_geom[0][i], ref_geom[1][j]])
                a = a * np.eye(ndim)
                psi0[i, j, target_state] = gwp(x, a=a, x0=[ref_geom[0][idx0], ref_geom[1][idx1]], ndim=2)
        # project it into the computational space using the overlap matrix
        psi0 = np.einsum('ijk,ij->ijk', A[:, :, :, idx0, idx1, target_state], psi0[:,:,target_state])
        return psi0



class LPA:
    def __init__(self, pes_data_file, nstates, save_file='A_approximation.npy'):
        with open(pes_data_file, 'rb') as f:
            self.pes_data = pickle.load(f)
        self.nx = int(np.sqrt(len(self.pes_data)))
        self.ny = self.nx
        self.nstates = nstates
        self.n_total = self.nx * self.ny * self.nstates
        self.save_file = save_file
        self.link = None

    def rebuild_molecule(self, mol_basis, mol_atom):
        mol = gto.Mole()
        mol.atom = mol_atom
        mol.basis = mol_basis
        mol.spin = 0
        mol.charge = 1
        mol.unit = 'bohr'
        mol.build()
        return mol

    def compute_overlap(self, data1, data2, state1, state2):

        mol1 = self.rebuild_molecule(data1['mol_basis'], data1['mol_atom'])
        mol2 = self.rebuild_molecule(data2['mol_basis'], data2['mol_atom'])
        mo_coeff1 = data1['mo_coeff']
        mo_coeff2 = data2['mo_coeff']

        s12 = gto.intor_cross('cint1e_ovlp_sph', mol1, mol2)
        s12 = reduce(np.dot, (mo_coeff1.T, s12, mo_coeff2))
        nmo = mo_coeff2.shape[1]
        nocc = mol2.nelectron // 2

        overlap = ci.cisd.overlap(data1['ci_vector'][state1], data2['ci_vector'][state2], nmo, nocc, s12)
        return overlap

    def calculate_link(self):
        self.link = np.zeros((self.nx, self.ny, self.nstates, self.nx, self.ny, self.nstates))

        for i in tqdm(range(self.nx)):
            for j in range(self.ny):
                for state1 in range(self.nstates):
                    for state2 in range(self.nstates):


                        # # Self overlap
                        # self.link[i, j, state1, i, j, state2] = self.compute_overlap(self.pes_data[i * self.ny + j], self.pes_data[i * self.ny + j], state1, state2)

                        # if 0 <= i - 1 < self.nx:
                        #     self.link[i - 1, j, state1, i, j, state2] = self.compute_overlap(self.pes_data[(i - 1) * self.ny + j], self.pes_data[i * self.ny + j], state1, state2)
                        #     self.link[i, j, state1, i - 1, j, state2] = self.compute_overlap(self.pes_data[i * self.ny + j], self.pes_data[(i - 1) * self.ny + j], state1, state2)

                        # if 0 <= j - 1 < self.ny:
                        #     self.link[i, j - 1, state1, i, j, state2] = self.compute_overlap(self.pes_data[i * self.ny + (j - 1)], self.pes_data[i * self.ny + j], state1, state2)
                        #     self.link[i, j, state1, i, j - 1, state2] = self.compute_overlap(self.pes_data[i * self.ny + j], self.pes_data[i * self.ny + (j - 1)], state1, state2)


                        # Self overlap
                        self.link[i, j, state1, i, j, state2] = self.compute_overlap(self.pes_data[i * self.ny + j], self.pes_data[i * self.ny + j], state1, state2)

                        # Neighbors in the diagonal directions
                        for delta in [-1, 1]:
                            if 0 <= i + delta < self.nx:
                                self.link[i + delta, j, state1, i, j, state2] = self.compute_overlap(self.pes_data[(i + delta) * self.ny + j], self.pes_data[i * self.ny + j], state1, state2)
                                self.link[i, j, state1, i + delta, j, state2] = self.compute_overlap(self.pes_data[i * self.ny + j], self.pes_data[(i + delta) * self.ny + j], state1, state2)

                            if 0 <= j + delta < self.ny:
                                self.link[i, j + delta, state1, i, j, state2] = self.compute_overlap(self.pes_data[i * self.ny + (j + delta)], self.pes_data[i * self.ny + j], state1, state2)
                                self.link[i, j, state1, i, j + delta, state2] = self.compute_overlap(self.pes_data[i * self.ny + j], self.pes_data[i * self.ny + (j + delta)], state1, state2)
        return self.link

    def save_overlaps(self, overlaps, filename):
        with open(filename, 'wb') as f:
            pickle.dump(overlaps, f)

    def _linked_product_approximation_2D(self, Lx, Ly):
        """原始方法1：linked_product_approximation_2D"""
        A1d = A2d = csr_matrix(np.zeros(Lx.shape))
        I = eye(Lx.shape[0], format="csr")

        A1d = Lx @ (I - Lx**(self.nx - 1)) @ inv(I - Lx)
        A1d = A1d + A1d.conj().T

        for j in range(1, self.ny):
            A2d += A1d @ Ly**j + Ly**j
        A2d = A2d + A2d.conj().T

        Atot = A1d + A2d
        print("Atot.shape:", Atot.shape)
        return Atot

    def _extract_neighbors(self, A):
        """提取最近邻与对角元信息，并构造成稀疏矩阵"""

        nearest_neighbor_x = np.zeros_like(A)
        nearest_neighbor_y = np.zeros_like(A)
        diagonal = np.zeros_like(A)

        for i in range(self.nx):
            for j in range(self.ny):
                diagonal[i, j, :, i, j, :] = A[i, j, :, i, j, :]
                if i + 1 < self.nx:
                    nearest_neighbor_x[i, j, :, i + 1, j, :] = A[i, j, :, i + 1, j, :]
                if j + 1 < self.ny:
                    nearest_neighbor_y[i, j, :, i, j + 1, :] = A[i, j, :, i, j + 1, :]

        self.Lx = csr_matrix(nearest_neighbor_x.reshape(self.n_total, self.n_total))
        self.Ly = csr_matrix(nearest_neighbor_y.reshape(self.n_total, self.n_total))
        self.A_diagonal = csr_matrix(diagonal.reshape(self.n_total, self.n_total))

    def _reshape_and_save(self, A_appro):
        A_appro = A_appro + self.A_diagonal
        A_appro = A_appro.toarray()
        A_appro = A_appro.reshape(self.nx, self.ny, self.nstates, self.nx, self.ny, self.nstates)
        np.save(self.save_file, A_appro)
        return A_appro

    def approximate_overlap(self):
        print('Step 1: Extracting neighbors and diagonals...')
        self._extract_neighbors(self.link)

        print(f'Step 2: Running linked product approximation ...')
        A_linked = self._linked_product_approximation_2D(self.Lx, self.Ly)

        print('Step 3: Reshaping and saving to file...')
        A_appro = self._reshape_and_save(A_linked)

        print(f'Done. Approximate A saved to: {self.save_file}')
        return A_appro


def initialize_wavepacket(ref_geom, idx0, idx1, a, A, target_state=1):
    """
    Initialize a Gaussian wavepacket psi0 centered at position x0.

    Parameters:
    - ref_geom: list of two 1D arrays [x_grid, y_grid]
    - nx, ny: grid sizes in x and y directions
    - nstates: number of electronic states
    - idx0, idx1: index of initial position (e.g. [0.0, 1.71875])
    - a: 2x2 matrix (numpy array), Gaussian width matrix
    - target_state: the index of the adiabatic state (default is 1)
    - A: global electronic overlap matrix

    Returns:
    - psi0: complex array of shape (nx, ny, nstates)
    """
    ndim = self.ndim
    nx, ny = len(ref_geom[0]), len(ref_geom[1])
    psi0 = np.zeros((nx, ny, self.nstates), dtype=complex)
    for i in range(nx):
        for j in range(ny):
            x = np.array([ref_geom[0][i], ref_geom[1][j]])
            a = a * np.eye(ndim)
            psi0[i, j, target_state] = gwp(x, a=a, x0=[ref_geom[0][idx0], ref_geom[1][idx1]], ndim=2)
    # project it into the computational space using the overlap matrix
    psi0 = np.einsum('ijk,ij->ijk', A[:, :, :, idx0, idx1, target_state], psi0[:,:,target_state])
    return psi0


class PyscfDriver:
    def __init__(self, atom, spin=0, nstates=3, basis='ccpvtz', method='cisd'):

        self.nstates = nstates
        self.basis = basis
        self.mol = None
        # self.nx = len(reference_geometry[0])
        # self.ny = len(reference_geometry[1])
        # self.x = reference_geometry[0]
        # self.y = reference_geometry[1]


        mol = gto.Mole(atom)
        mol.basis = basis
        mol.spin = spin  # 2 * S, where S = 0 for a singlet state with 2 electrons
        mol.charge = 1  # +1 charge for H3+ system with 2 electrons
        mol.unit = 'bohr'  # Set unit to Bohr
        mol.build()

        self.mol = mol
        self.ci = None

        return

    def run(self):
        if self.method == 'cisd':
            return self.cisd()
        else:
            raise ValueError('Method {} is not supported. Try cisd.'.format(self.method))

    def cisd(self):

        # self.mol.set_geom_(R) # update geometry

        mf = scf.RHF(self.mol)
        # mf.init_guess = 'atom'  # Use 'atom' initial guess method
        mf.kernel()

        myci = ci.CISD(mf)
        myci.nstates = self.nstates
        myci.run()

        self.ci = myci
        self.e_tot = myci.e_tot
        self.ci = myci.ci

        return myci

        # return {
        #     'e_s0': myci.e_tot[0],
        #     'e_s1': myci.e_tot[1],
        #     'e_s2': myci.e_tot[2],
        #     'mo_coeff': mf.mo_coeff,
        #     'mo_energy': mf.mo_energy,
        #     'mol_basis': mf.mol.basis,
        #     'mol_atom': mf.mol.atom,
        #     'ci_vector': myci.ci
        # }


def overlap(ci_list, nstates, dtype=float):

    # mol1 = self.rebuild_molecule(data1['mol_basis'], data1['mol_atom'])
    # mol2 = self.rebuild_molecule(data2['mol_basis'], data2['mol_atom'])

    # mo_coeff1 = data1['mo_coeff']

    N = len(ci_list)

    A = np.zeros((N, N, nstates, nstates), dtype=dtype)

    for n in range(N):
        A[n,n] = eye(nstates)

    for n in range(N):

        cibra = ci_list[n]
        mo_coeff1 = cibra.mo_coeff
        mol1 = cibra.mol
        ci1 = cibra.ci

        for m in range(n):
            ciket = ci_list[m]
            mo_coeff2 = ciket.mo_coeff
            mol2 = ciket.mol


        # ci1 = self.ci

    # mo_coeff2 = data2['mo_coeff']


    s12 = gto.intor_cross('cint1e_ovlp_sph', mol1, mol2)
    s12 = reduce(np.dot, (mo_coeff1.T, s12, mo_coeff2))
    nmo = mo_coeff2.shape[1]
    nocc = mol2.nelectron // 2

    nstates = self.nstates

    S = np.zeros((nstates, nstates))

    # for i in range(1, nstates):
    #     S[i-1, i-1] = ci.cisd.overlap(ci1.ci[i-1], ci2.ci[i-1], nmo, nocc)

    # print('<CISD-mol1|CISD-mol2> = ',
    for i in range(1, nstates):
        for j in range(1, i):
            S[i-1, j-1] = overlap(ci1.ci[i-1], ci2.ci[j-1], nmo, nocc)
            S[j-1, i-1] = S[i-1, j-1]
    return S

    for n in range(self.nstates):
        for m in range(n):
            overlap[n,m] = ci.cisd.overlap(self.ci[n], ci2[m], nmo, nocc, s12)

    return overlap

# def overlap(cibra, ciket):

#     # mol1 = self.rebuild_molecule(data1['mol_basis'], data1['mol_atom'])
#     # mol2 = self.rebuild_molecule(data2['mol_basis'], data2['mol_atom'])

#     # mo_coeff1 = data1['mo_coeff']

#     mo_coeff1 = cibra.mo_coeff
#     ci1 = self.ci

#     # mo_coeff2 = data2['mo_coeff']


#     s12 = gto.intor_cross('cint1e_ovlp_sph', mol1, mol2)
#     s12 = reduce(np.dot, (mo_coeff1.T, s12, mo_coeff2))
#     nmo = mo_coeff2.shape[1]
#     nocc = mol2.nelectron // 2

#     nstates = self.nstates

#     S = np.zeros((nstates, nstates))

#     for i in range(1, nstates):
#         S[i-1, i-1] = ci.cisd.overlap(ci1.ci[i-1], ci2.ci[i-1], nmo, nocc)

#     # print('<CISD-mol1|CISD-mol2> = ',
#     for i in range(1, nstates):
#         for j in range(1, i):
#             S[i-1, j-1] = overlap(ci1.ci[i-1], ci2.ci[j-1], nmo, nocc)
#             S[j-1, i-1] = S[i-1, j-1]
#     return S

#     # for n in range(self.nstates):
#     #     for m in range(n):
#     #         overlap[n,m] = ci.cisd.overlap(self.ci[n], ci2[m], nmo, nocc, s12)

#     return overlap

if __name__=='__main__':
    from pyqed.phys import gwp
    from pyqed.units import au2fs


    ndim = 2 # Number of nuclear degrees of freedom
    nstates = 3 # Number of electronic states
    mass = [1836, 1836] # Nuclear mass


    domains = [[-5.5, 5.5], [0, 11]] # Nuclear coordinate domains
    levels = [6,6] # Discretization levels

    ldr = LDRN(domains=domains, levels=levels, ndim=ndim, nstates=nstates, mass=mass)


    psi0 = initialize_wavepacket(ref_geom=ref_geom, idx0=32, idx1=10, a=30, A=overlap_matrix, target_state=1) # x[idx0], y[idx1]是波包初始位置的中心点


    def transform2Cartesian_coord(domains, levels):
        reference_geometries = []
        for d in range(len(domains)):
            reference_geometries.append(discretize(*domains[d], levels[d], endpoints=True))
        return reference_geometries
    ref_geom = transform2Cartesian_coord(domains, levels)

    def coord_transform(x, t=None, x0=None):
        """
        transform from reactive coords to Cartesian coords for quantum chemistry

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        t : 2darray
            transformation matrix of shape (3Natom, d)

        Returns
        -------
        None.

        """
        R = x  + [0]
        return R

    l = 2
    left = np.array([-l / 2, 0, 0])
    right = np.array([ l/2, 0, 0])
    R = [0, 0, 0]

    atom = [['H', left], ['H', right], ['H', R]]




     ### quantum chemistry
    driver = Driver(mol, nstates=nstates, basis='ccpvtz', method='cisd', engine='PySCF')
    # apes = mol.scan_pes()

    x, y = ldr.x
    nx, ny = len(x), len(y)

    # build the adiabatic potential energy

    v = np.zeros((nx, ny, nstates))
    pes_data = []

    for i in range(nx):
        for j in range(ny):

            q = [x[i], y[j]] # reactive coordinates
            R = coord_transform(q) # transform to Cartesian coordinates

            result = driver(R)

            v[i, j] = e_tot

            # APES[i, j, 0] = result['e_s0']
            # APES[i, j, 1] = result['e_s1']
            # APES[i, j, 2] = result['e_s2']

            pes_data.append(result)  # Store all energy states for the configuration

    # with open('qchem_data_total.pkl', 'wb') as f:
    #     pickle.dump(pes_data, f)

    E0_min = np.nanmin(v[:, :, 0])
    v = v - E0_min


    # build the global electronic overlap matrix

    lpa = LPA('qchem_data_total.pkl', nstates=nstates)

    link = lpa.calculate_link()

    overlap_matrix = lpa.approximate_overlap()


  #psi0 = ldr.initialize_wavepacket(ref_geom=ref_geom, idx0=0, idx1=0, a=10, A=overlap_matrix, target_state=1)



    # load electronic structure data into the quantum dynamics solver

    ldr.apes = v
    ldr.A = overlap_matrix
    result = ldr.run(psi0, dt = 0.00390625/au2fs, nt = 7680)

    result.dump('result')