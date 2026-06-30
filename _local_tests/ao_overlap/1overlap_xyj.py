import numpy as np
from pyqed.qchem.gdvr import AtomicChain
from pyqed.qchem.gdvr.integrals import overlap_2d_cartesian, build_h1_nm, V_en_sp_total_at_z
from pyqed.qchem.gdvr.rhf import SweepNewtonHelper, sweep_optimize_driver, rebuild_Hcore_from_d, eri_JK_from_kernels_M1
from pyqed.qchem.gdvr.newton import CollocatedERIOp, NewtonHelper

def build_xy_overlap_Sprim(molA, molB):

    alA  = molA._gdvr_build_context["alphas"]
    cenA = molA._gdvr_build_context["centers"]
    labA = molA._gdvr_build_context["labels"]

    alB  = molB._gdvr_build_context["alphas"]
    cenB = molB._gdvr_build_context["centers"]
    labB = molB._gdvr_build_context["labels"]

    nA = len(alA)
    nB = len(alB)

    Sxy = np.zeros((nA,nB))

    for i in range(nA):
        for j in range(nB):

            Sij = overlap_2d_cartesian(np.array([alA[i], alB[j]]), np.array([cenA[i], cenB[j]]),[labA[i], labB[j]])[0,1]

            Sxy[i,j] = Sij

    return Sxy


def gdvr_ao_overlap_from_dstack(ao1, ao2, Sprim12):
    """
    overlap between optimized GDVR orbitals.

    Parameters
    ----------
    ao1 : (Nz,nAO)
    ao2 : (Nz,nAO)

    Sprim12 : (nAO,nAO)

    Returns
    -------
    Sgdvr : (Nz*M, Nz*M)
    """
    # return ao1.conj() @ Sprim12 @ ao2.T
    vals = np.einsum("ni,ij,nj->n", ao1.conj(), Sprim12, ao2, optimize=True)
    return np.diag(vals)


def rhf_overlap(mfA, aoA, molA, mfB, aoB, molB):

    Sprimitive = build_xy_overlap_Sprim(molA,molB)
    Sao = gdvr_ao_overlap_from_dstack(aoA, aoB, Sprimitive)

    nocc = molA.nelec // 2
    CA = mfA.mo_coeff[:, :nocc]
    CB = mfB.mo_coeff[:, :nocc]
    Socc = CA.conj().T @ Sao @ CB

    return np.linalg.det(Socc)

if __name__ == "__main__":

    coordsA = [(0,0,z) for z in [-2.1,-0.7,0.7,2.1]]
    molA = AtomicChain(["H"]*4, coordsA)
    molA.build(Lz=14.0,Nz=48,M=1,verbose=False)
    mfA = molA.RHF().run(conv=1e-8,verbose=False)
    _, aoA = mfA.newton(max_cycles=12,sweep_iterations=1,tol=1e-6,ridge=0.5,trust_step=0.5,trust_radius=1.0,verbose=True)     
    print(mfA.info["newton_converged"], '111111111111')


    coordsB = [(0,0,z) for z in [-2.1,-0.7,0.7,2.2]]
    molB = AtomicChain(["H"]*4, coordsB)
    molB.build(Lz=14.0,Nz=48,M=1,verbose=False)
    mfB = molB.RHF().run(conv=1e-8,verbose=False)
    _, aoB = mfB.newton(max_cycles=100,sweep_iterations=1,tol=1e-6,ridge=0.5,trust_step=0.5,trust_radius=1.0,verbose=True)  
    print(mfB.info["newton_converged"], '111111111111') #输出结果True 111111111111


    S_primitive = build_xy_overlap_Sprim(molA, molB) #原始高斯基组之间的重叠矩阵
    S_ao = gdvr_ao_overlap_from_dstack(aoA, aoB, S_primitive) #不同分子构型之间的原子轨道重叠矩阵

    nocc = molA.nelec // 2
    CA = mfA.mo_coeff[:, :nocc]
    CB = mfB.mo_coeff[:, :nocc]
    Socc1 = CA.conj().T @ S_ao @ CB
    det1 = np.linalg.det(Socc1)
    print(det1)

    Socc2 = CA.conj().T @ CB
    det2 = np.linalg.det(Socc2)
    print(det2)    

    U, s, Vh = np.linalg.svd(CA.T @ S_ao @ CB)
    U_align = U @ Vh
    CB_aligned = CB @ U_align
    Socc = CA.T @ S_ao @ CB_aligned
    det = np.linalg.det(Socc)
    print(det)
    
    # S = np.zeros(48)
    # for i in range(48):
    #     S[i] = aoA[i] @ S_primitive @ aoB[i]
    # print(S)
    # print(np.diag(S_ao))              # 是否接近 ±1
    # print(np.max(np.abs(S_ao - np.eye(48))))

    # nocc = molA.nelec // 2
    # CA = mfA.mo_coeff[:, :nocc]
    # CB = mfB.mo_coeff[:, :nocc]
    # Socc = CA.conj().T @ CB
    # det = np.linalg.det(Socc)
    # print(Socc)
    # print(det)

    # S_primitive.shape=(Ngauss, Ngauss)
    # aoA.shape=(Nz*M, Ngauss)
    # S_ao=(Nz*M, Nz*M)
    # ov_mo.shape=(Nao,Nao)
