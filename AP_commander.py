# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 18:56:07 2023

@author: vsots
"""

import numpy as np
import pandas as pd
import mouse_class as mcl
import draw_cells as drcl
import pc_io as pio
import scipy.ndimage as sf
from scipy.stats import pearsonr, chisquare, kstest
from scipy.stats import median_abs_deviation as mad
import scipy.interpolate as interp
from scipy import spatial
import matplotlib.pyplot as plt
import SpikingPasses as spps
import PlaceFields as pcf
import matplotlib.patches as patches
import os
import cmath
from glob import glob


root = 'F:\\VOVA\\APTSD\\'
names = ['AP_11', 'AP_12', 'AP_13', 'AP_14', 'AP_15']
days = ['0TR', '1CT', '2ST', '3GT', '4EM']
fps = 30

for name in names:
    for d in days:
        ms = mcl.EasyMouse(name + '_' + d)
        tstamp = np.genfromtxt(root + 'min1pipe\\' + name + '_' + d + '_timestamp.csv', delimiter = ',')
        tstamp = (tstamp[:,1] - tstamp[0,1])/1000
        neur = np.genfromtxt(root + 'min1pipe\\' + name + '_' + d + '_NR_data_processed_traces.csv', delimiter = ',')
        spikes = np.genfromtxt(root + 'min1pipe\\' + 'spikes_' + name + '_' + d + '_NR_data_processed_traces.csv', delimiter = ',')
        motind = np.genfromtxt(root + 'freezing\\' + name + '_' + d + '_MotionIndext_13_5_30_15.csv', delimiter = ',')
        freeze = np.genfromtxt(root + 'freezing\\' + name + '_' + d + '_Freezing_13_5_30_15.csv', delimiter = ',')
        ms.get_any_trace('time', tstamp)
        ms.get_any_trace('neur', neur[:, 1:])
        ms.get_any_trace('spikes', spikes[:, 1:])
        ms.get_any_trace('motind', motind, timestamp = np.linspace(0, len(motind)/fps, len(motind)))
        ms.get_any_trace('freeze', freeze, timestamp = np.linspace(0, len(freeze)/fps, len(freeze)))        
        np.savez(root + name + '_' + d + '.npz', [ms])
        
    