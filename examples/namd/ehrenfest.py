#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 10:28:26 2025

Ehrenfest dynamcis with Shin-Metiu moddel 

@author: Bing Gu (gubing@westlake.edu.cn)
"""

from pyqed.models.ShinMetiu import ShinMetiu2
import numpy as np

from pyqed.namd.mf import Ehrenfest

from pyqed import proton_mass as mp


mol = ShinMetiu2()

mol.build(domain=[[-10, 10], ] * 2, npts=[31, 31])

ed = Ehrenfest(ndim=mol.ndim, ntraj=1, nstates=mol.nstates, mass=[mp, ] * 2)

ed.nac_driver = mol.nonadiabatic_coupling

ed.sample(init_state=2, x0=[0, 1.3], ax=18)
ed.run(dt=0.5, nt=400, nout=2)

rho = ed.rdm()





# # slice the electronic data to see if it makes sense
# x = np.linspace(-2,2, 50)
# nx = len(x)
# e = np.zeros((nx, mol.nstates))
# grad = np.zeros((nx, mol.nstates, mol.ndim))
# nac = np.zeros((nx, mol.nstates, mol.nstates, mol.ndim))

# for i in range(len(x)):
#     R = [x[i], 0]
    
#     e[i], grad[i], nac[i] = mol.nonadiabatic_coupling(R)


# import ultraplot as plt
# fig, ax = plt.subplots()

# for i in range(mol.nstates):
#     ax.plot(x, e[:,i])

# fig, ax = plt.subplots()
# for i in range(mol.nstates):
#     ax.plot(x, grad[:,i, 0])
#     ax.plot(x, grad[:,i, 1])

