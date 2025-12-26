import numpy as np
import time
import scipy.linalg as la
from pyqed.qchem.dvr.hybrid_gauss_dvr_integrals_add_p_orbitals import (
    PrimitiveLabel,
    make_xy_sp_primitive_basis,
    overlap_2d_cartesian,
    kinetic_2d_cartesian,
    V_en_sp_total_at_z,
    V_en_2d_cartesian_single_nucleus,
    eri_2d_cartesian_with_p,
    pair_params,
    ao_K_of_delta,
    _permute_K_ikjl
)

# Dummy class to mimic your Molecule structure
class Molecule:
    def __init__(self, charges, coords, nelec=None):
        self.charges = np.asarray(charges, float).reshape(-1)
        self.coords  = np.asarray(coords,  float).reshape(-1, 3)
        self.nelec = int(round(np.sum(self.charges))) if nelec is None else int(nelec)
    def to_tuples(self):
        return [(float(Z), float(x), float(y), float(z))
                for (Z, (x, y, z)) in zip(self.charges, self.coords)]
    def nuclear_repulsion_energy(self):
        E = 0.0
        Z, R = self.charges, self.coords
        for i in range(len(Z)):
            for j in range(i + 1, len(Z)):
                E += Z[i] * Z[j] / np.linalg.norm(R[i] - R[j])
        return E

def test_ven_derivatives():
    """
    Strict Finite Difference Check for V_en integrals.
    FIXED: Now uses distinct centers to isolate bra/ket derivatives.
    """
    print("\n" + "="*60)
    print("  DIAGNOSTIC 1: V_en Finite Difference Check")
    print("="*60)
    
    # Setup: nucleus at origin
    alpha = 1.5
    center = np.array([0.5, 0.0]) 
    nuc_xy = np.array([0.0, 0.0])
    dz = 0.0 
    
    # Labels
    lbl_s  = [PrimitiveLabel('2d-s', 2, (0,0,0))]
    lbl_px = [PrimitiveLabel('2d-px', 2, (1,0,0))]
    
    # 1. Analytical Value <px(A)|V|s(A)>
    # We compute the 1-center block
    alphas_1 = np.array([alpha, alpha])
    centers_1 = np.array([center, center])
    labels_1 = lbl_s + lbl_px
    
    V_ana = V_en_2d_cartesian_single_nucleus(alphas_1, centers_1, labels_1, nuc_xy, dz)
    val_ps_ana = V_ana[1,0] # <px|V|s>
    
    print(f"Analytical <px|V|s>: {val_ps_ana: .10e}")
    
    # 2. Finite Difference
    # To test <px|V|s> = (1/2a) d/dAx <s|V|s>, we must shift ONLY the bra center.
    # Construct a 2-center system: Center A (shifted), Center B (fixed at original).
    # We look at element [0, 1] -> <s(A)|V|s(B)>
    
    h = 1e-5
    alphas_2 = np.array([alpha, alpha])
    labels_2 = lbl_s + lbl_s # s on A, s on B
    
    # Forward: A = center + h
    c_plus = np.array([center.copy(), center.copy()])
    c_plus[0, 0] += h
    V_plus = V_en_2d_cartesian_single_nucleus(alphas_2, c_plus, labels_2, nuc_xy, dz)[0,1]
    
    # Backward: A = center - h
    c_minus = np.array([center.copy(), center.copy()])
    c_minus[0, 0] -= h
    V_minus = V_en_2d_cartesian_single_nucleus(alphas_2, c_minus, labels_2, nuc_xy, dz)[0,1]
    
    dV_dx = (V_plus - V_minus) / (2*h)
    val_ps_num = (1.0 / (2*alpha)) * dV_dx
    
    print(f"Numerical  <px|V|s>: {val_ps_num: .10e}")
    
    err = abs(val_ps_ana - val_ps_num)
    print(f"Error:      {err: .2e}  {'[PASS]' if err < 1e-6 else '[FAIL]'}")
    
    # 3. p-p check
    # <px|V|px> matches (1/2a) d/dBx <px(A)|V|s(B)>
    # Center A fixed (px), Center B shifts (s)
    val_pp_ana = V_ana[1,1]
    
    labels_mix = lbl_px + lbl_s
    
    c_b_plus = np.array([center.copy(), center.copy()])
    c_b_plus[1, 0] += h
    V_pp_plus = V_en_2d_cartesian_single_nucleus(alphas_2, c_b_plus, labels_mix, nuc_xy, dz)[0,1]
    
    c_b_minus = np.array([center.copy(), center.copy()])
    c_b_minus[1, 0] -= h
    V_pp_minus = V_en_2d_cartesian_single_nucleus(alphas_2, c_b_minus, labels_mix, nuc_xy, dz)[0,1]
    
    dVps_dx = (V_pp_plus - V_pp_minus) / (2*h)
    val_pp_num = (1.0 / (2*alpha)) * dVps_dx
    
    print(f"\nCheck <px|V|px>:")
    print(f"  Analytical: {val_pp_ana: .10e}")
    print(f"  Numerical:  {val_pp_num: .10e}")
    err_pp = abs(val_pp_ana - val_pp_num)
    print(f"  Error:      {err_pp: .2e}  {'[PASS]' if err_pp < 1e-5 else '[FAIL]'}")


def analyze_energy_components():
    print("\n" + "="*60)
    print("  DIAGNOSTIC 2: Energy Component Breakdown")
    print("="*60)

    charges = np.array([1.0, 1.0, 1.0, 1.0])
    coords = np.array([
        # [0.0, 0.0,  3.6 ],
        # [0.0, 0.0,  0.91],
        # [0.0, 0.0, -3.6 ],
        # [0.0, 0.0, -0.91],
        [0.0,  0.7,  0.7],
        [0.0,  0.7, -0.7],
        [0.0, -0.7,  0.7],
        [0.0, -0.7, -0.7],
    ])
    mol = Molecule(charges, coords)
    
    from pyqed.qchem.dvr.hybrid_gauss_dvr_integrals_add_p_orbitals import Exp_631g_ss_H
    s_exps = Exp_631g_ss_H
    p_exps = np.array([1.1]) 

    nuclei = mol.to_tuples()
    alphas, centers, labels = make_xy_sp_primitive_basis(nuclei, s_exps, p_exps)
    
    print(f"Basis Size: {len(alphas)} (s + p)")
    
    S = overlap_2d_cartesian(alphas, centers, labels)
    T = kinetic_2d_cartesian(alphas, centers, labels)
    
    # V_en at a nucleus slice
    z_slice = 0.91
    z_slice = 0.92
    V = V_en_sp_total_at_z(alphas, centers, labels, nuclei, z_slice)
    
    H = T + V
    
    # Diagonalize
    w, U = np.linalg.eigh(S)
    X = U @ np.diag(1.0/np.sqrt(w)) @ U.T
    Hp = X.T @ H @ X
    e, C_p = np.linalg.eigh(Hp)
    C = X @ C_p
    
    nocc = 2
    C_occ = C[:, :nocc]
    P = 2.0 * (C_occ @ C_occ.T)
    
    E_kin = np.sum(P * T)
    E_nuc_att = np.sum(P * V)
    
    print(f"\n--- One-Electron Energy (at z={z_slice}) ---")
    print(f"Kinetic <T>:       {E_kin:.6f} Eh")
    print(f"Attraction <V_en>: {E_nuc_att:.6f} Eh")
    
    # 2D ERI
    eri = eri_2d_cartesian_with_p(alphas, centers, labels, delta_z=0.015)
    J_val = np.einsum('pqrs,pq,rs->', eri, P, P)
    K_val = np.einsum('pqrs,ps,rq->', eri, P, P)
    
    E_coul = 0.5 * J_val
    E_exch = -0.25 * K_val
    E_2e = E_coul + E_exch
    
    print(f"Coulomb <J>:       {E_coul:.6f} Eh")
    print(f"Exchange <K>:      {E_exch:.6f} Eh")
    
    E_tot_slice = E_kin + E_nuc_att + E_2e
    print(f"\nTotal Slice Energy (approx): {E_tot_slice:.6f} Eh")

def analyze_energy_components_pyscf():
    """
    Reference energy component breakdown with PySCF using 6-31G** basis
    for the same H4 geometry as in analyze_energy_components().
    """
    print("\n" + "="*60)
    print("  PySCF REFERENCE: 6-31G** Energy Components")
    print("="*60)

    from pyscf import gto, scf

    # Same geometry as above (assumed in Bohr)
    charges = np.array([1.0, 1.0, 1.0, 1.0])
    coords = np.array([
        [0.0, 0.0,  3.6 ],
        [0.0, 0.0,  0.91],
        [0.0, 0.0, -3.6 ],
        [0.0, 0.0, -0.91],
    ])

    # Build PySCF molecule
    atom_spec = []
    for Z, (x, y, z) in zip(charges, coords):
        # All are hydrogens in this test
        atom_spec.append(f"H {x:.10f} {y:.10f} {z:.10f}")

    mol_pyscf = gto.Mole()
    mol_pyscf.atom  = "; ".join(atom_spec)
    mol_pyscf.unit  = "Bohr"     # use same units as your DVR code
    mol_pyscf.basis = "6-31g**"
    mol_pyscf.charge = 0
    mol_pyscf.spin   = 0         # 2S = N_alpha - N_beta = 0 for closed-shell H4
    mol_pyscf.build()

    # RHF calculation
    mf = scf.RHF(mol_pyscf)
    mf.verbose = 0
    E_hf = mf.kernel()

    # 1e and 2e integrals / density
    hcore = mf.get_hcore()
    dm = mf.make_rdm1()

    # Split hcore into kinetic and nuclear attraction
    T_ao = mol_pyscf.intor("int1e_kin")
    V_en_ao = hcore - T_ao

    # J and K matrices in AO basis
    J_ao, K_ao = mf.get_jk(mol_pyscf, dm)

    # Energy components in Hartree
    E_kin     = np.einsum("ij,ji->", T_ao, dm)
    E_nuc_att = np.einsum("ij,ji->", V_en_ao, dm)
    E_coul    = 0.5  * np.einsum("ij,ji->", J_ao, dm)
    E_exch    = -0.25 * np.einsum("ij,ji->", K_ao, dm)
    E_nuc_rep = mol_pyscf.energy_nuc()
    E_tot     = E_kin + E_nuc_att + E_coul + E_exch + E_nuc_rep

    print("\n--- One-Electron Energy (PySCF, 6-31G**) ---")
    print(f"Kinetic <T>:           {E_kin: .8f} Eh")
    print(f"Attraction <V_en>:     {E_nuc_att: .8f} Eh")

    print("\n--- Two-Electron Energy (PySCF, 6-31G**) ---")
    print(f"Coulomb <J>:           {E_coul: .8f} Eh")
    print(f"Exchange <K>:          {E_exch: .8f} Eh")

    print("\n--- Nuclear Repulsion and Total ---")
    print(f"Nuclear Repulsion:     {E_nuc_rep: .8f} Eh")
    print(f"HF Total Energy:       {E_hf: .8f} Eh")
    print(f"Reconstructed Total:   {E_tot: .8f} Eh")




if __name__ == "__main__":
    test_ven_derivatives()
    analyze_energy_components()
    # analyze_energy_components_pyscf()