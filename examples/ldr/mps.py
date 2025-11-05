#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 10:10:32 2025

Compute the nuclear gradients  of the electronic Hamiltonian

@author: Bing Gu (gubing@westlake.edu.cn)
"""
from pyscf import gto

mol = gto.M(atom='H 0,0,0; H 0,0,1', basis='sto6g')

F = 0 # <\nabla |nuc| >
for atm_id in range(mol.natm):
    with mol.with_rinv_at_nucleus(atm_id):
        vrinv = mol.intor('int1e_iprinv', comp=3) # <\nabla |nuc| >
        vrinv *= -mol.atom_charge(atm_id)
        
        F += vrinv + vrinv.transpose(0,2,1)
    
print(F)



