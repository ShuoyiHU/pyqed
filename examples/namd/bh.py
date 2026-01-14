#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 09:13:59 2025

@author: Bing Gu (gubing@westlake.edu.cn)
"""

from pyqed.namd import BornHuang2
import numpy as np
import time

#print('For two-photon excited wavepacket \n')
ndim = 2
nstates = 2 # total number of electronic states
 # setup the initial wavefunction

psi0 = ...


# idx = np.unravel_index(np.argmin(np.abs(s0)), s0.shape)
# print(s0[idx], s1[idx], s2[idx])
# gap20 = (s2[idx] - s0[idx])
# gap10 = s1[idx] - s0[idx]
# print('vertical transition energy 0-2 = {} eV'.format(gap20 * au2ev))
# print('vertical transition energy 0-1 = {} eV'.format(gap10 * au2ev))

# setup grid
nx = 2 ** 6
ny = 2 ** 6
xmin = -8
xmax = -xmin
ymin = -8
ymax = -ymin
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
# dx = q1[1] - q1[0]
# dy = q2[1] - q2[0]

# nx = len(q1)
# ny = len(q2)

# X, Y = np.meshgrid(q1, q2)

# setup Fourier space
# kx = 2. * np.pi * fftfreq(nx, dx)
# ky = 2. * np.pi * fftfreq(ny, dy)

# print('torsion range = ', q1[0], q1[-1])
# print('dq1 = {}'.format(dx))
# print('number of grid points along x = {} \n'.format(nx))
# print('CNN-NNC bending range = ', q2[0], q2[-1])
# print('dq2 = {}'.format(dy))
# print('number of grid points along y = {}'.format(ny))


# metric tensor
# G = np.zeros((nx, ny, ndim, ndim))
# G[:,:, 0, 0] = g11
# G[:, :, 0, 1] = G[:, :, 1, 0] = g12
# G[:, :, 1, 1] = g22

# fig, (ax1, ax2) = subplots(ncols=2)
# im1 = ax1.imshow(g11)
# fig.colorbar(im1)
# ax2.imshow(g22)
# ax1.set_title('G matrix')

# sigma = np.identity(2) * 2.
# # nuclear wavepackets in adiabatic surfaces
# psi0 = np.zeros((nx, ny, nstates), dtype=complex)
# coeff1, phase = np.sqrt(0.), 0
# coeff2 = 1.

# x0, y0, kx0, ky0 = 1., 1.0, 0.0, 0
# psi0[:,:,0] = coeff1 * gauss_x_2d(X, Y, sigma, x0, y0, kx0, ky0)
#                  * np.exp(1j*phase)
# psi0[:,:,1] = coeff2 * gauss_x_2d(X, Y, sigma, x0, y0, kx0, ky0)



# ######
# t0 = -10/au2fs
# t = t0

# # set/read APESs & TDM & NACs

# vmat = APES(X, Y)
# tdm = np.zeros((nx, ny, nstates, nstates))
# nac = np.zeros((nx, ny, ndim))



# We have assumed a frozen approxmation for the tdm02*psi0 on S2 PES
# for time t2 - t1

# here we can simply run adiabatic dynamics on S1 surface
# take the wavepacket to S1 by TDM and propagate to final time tf
# tf should be large enough so that the field vanishes

# specify time steps and duration (atomic units are used throughout)
dt = 0.2
nt = 100



start_time = time.time()


bh = BornHuang2(x, y)

bh.run(psi0, dt=dt, nt=nt)



print('Execution Time = {} s'.format(time.time() - start_time))