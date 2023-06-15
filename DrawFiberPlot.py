# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 14:19:09 2021

@author: vsots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def DrawFiberPlot(x, y, neur, Nbins = 5):
    
    neuro_bin = np.zeros(neur.shape, dtype = int)
    neuro_bin = (neur - np.min(neur))*Nbins/(np.max(neur) - np.min(neur))
    neuro_bin[neuro_bin == Nbins] -= 1
        
    for n_bin in range(Nbins):
        color = cm.jet(n_bin/Nbins)
        plt.plot(x[neuro_bin == n_bin], y[neuro_bin == n_bin], color = color, ls = ' ', ms=2, marker='.', zorder = 0.0)
    return
        
        