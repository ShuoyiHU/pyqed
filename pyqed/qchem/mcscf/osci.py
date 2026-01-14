#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 09:53:43 2026

OSCI 

@author: 
    Junzhe Zhang (zhangjunzhe@westlake.edu.cn) 
    Bing Gu (gubing@westlake.edu.cn)
"""


class OSCI:
    def __init__(self, sci):
        pass
    
    def run():
        pass


if __name__ == '__main__':
    
    from pyqed import Molecule 
    from pyqed.qchem.mcscf.casci import CASCI 
    
    mol = Molecule(atom = [
    ['Li' , (0. , 0. , 0)],
    ['H' , (0. , 0. , 1.4)], ])
    mol.basis = 'ccpvdz'

    # mol.unit = 'b'
    mol.build(driver='pyscf')

    mf = mol.RHF().run()


    ncas, nelecas = (4,2)
    mc = CASCI(mf, ncas, nelecas)
    mc.run(2)