# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 17:30:32 2022

@author: vsots
"""

import cv2
import matplotlib.pyplot as plt
import glob

root = 'F:\FAD_mice'
date = '2022_09_21'
outfname = root + '\\mice_screenshots.png'

fig = plt.figure(figsize=(40,30), dpi=100)
plt.rcParams['font.size'] = '16'

for i,name in enumerate(glob.glob(root + '\*\\' + date + '\*\Miniscope\\0.avi')):
    cap = cv2.VideoCapture(name)
    ret, frame = cap.read()
    ax = plt.subplot(3,4,i+1)
    ax.title.set_text(name[len(root)+1:len(root)+5])
    ax.imshow(frame)
    cap.release()
    
plt.savefig(outfname)
