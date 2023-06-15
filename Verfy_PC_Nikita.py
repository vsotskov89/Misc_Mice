# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:16:21 2020

@author: VOVA
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import kstest
from scipy.stats import uniform
 
def change_labels_color(ax):
    [t.set_color('white') for t in ax.xaxis.get_ticklines()]
    [t.set_color('white') for t in ax.xaxis.get_ticklabels()]
    [t.set_color('white') for t in ax.yaxis.get_ticklines()]
    [t.set_color('white') for t in ax.yaxis.get_ticklabels()]

n = 20 # number of points
significance_lvl = 0.01
r = range(n)
rand_angle = [2*np.pi*random.random() for _ in range(n)]
non_rand_angle = [2*np.pi*random.uniform(0,0.2) for _ in range(n)]

fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection='polar'), figsize = (24,12))
fig.suptitle('Random vs. non-random distribution')
change_labels_color(ax1)
change_labels_color(ax2)
ax1.plot(rand_angle, r, 'ro')
ax2.plot(non_rand_angle, r, 'ro')

nbins = 50
bins = np.linspace(0, 2*np.pi, nbins)
fig2, ax = plt.subplots(figsize = (24,12))
change_labels_color(ax)
plt.hist(rand_angle, bins, alpha=0.5, label='x')
plt.hist(non_rand_angle, bins, alpha=0.5, label='y')
plt.legend(loc='upper right')
plt.show()

for data in [rand_angle, non_rand_angle]:
    stat, p = kstest(data, 'uniform', N = n, args = (0.0, 2*np.pi))
    print('KS statistics:', stat)
    print('p-value:', p)
    if p < significance_lvl:
        result = 'Place cell'
    else:
        result = 'Noise cell'
    print(result)
    print('--------------------')
