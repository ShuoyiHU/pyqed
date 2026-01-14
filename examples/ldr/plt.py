#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 15:52:01 2025

@author: bingg
"""

import numpy as np

E = np.load('energy.npz')['arr_0']

import ultraplot as plt

fig, ax = plt.subplots()
ax.plot(E[1], '-o')
ax.plot(E[2], '-s')

# ax.format(ylim=(-261.08, -261.0))