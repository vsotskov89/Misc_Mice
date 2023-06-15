# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 00:04:46 2022

@author: vsots
"""

import cv2
import numpy as np
import pc_io as pio
import mouse_class as mcl
import draw_cells as drcl
import os

path_npz = 'D:\Work\\NEW_MMMICE\CA1_25_1D_nnew.npz'
path_neur = 'D:\Work\\PC_video\\neur.avi'
path_neur_out = 'D:\Work\\PC_video\\neur_merged.mp4'
path_filters = 'D:\Work\\PC_video\\filters\\'
path_mask = 'D:\Work\\PC_video\\MAX_contours.tif'
path_beh = 'D:\Work\\PLACE CELLS\\CA1_25_1D.m4v'
path_beh_out = 'D:\Work\\PC_video\\behavior_output.mp4'
path_out = 'D:\Work\\PC_video\\result.mp4'
nums = np.array([45,125,153,183,194,197,207,268,286,303])-1 #1-based!!!
lay = [0,0,1,2,1,0,2,1,3,3]

t_start = 67*20
t_len = 3000
center = (690,550)
rad = 270
drad = 25
gap = 5

def colornum_Metro(num):#BGR
    #Returns color for each number as in Moscow Metro
    return {
    1:[0,0,1],        #red
    2:[0,0.7,0],      #green
    3:[1,0,0],        #blue
    4:[1,1,0],        #cyan
    5:[0.2,0.25,0.5], #brown
    6:[0.1,0.5,1],    #orange
    7:[1,0,0.5],      #violet
    8:[0,0.7,0.8],    #yellow
    9:[0.8,0.7,1],  #pink
    0:[0.1,1,0.5]}.get(num%10)   #lime

#%% Draw place fields on behavior video
#drop first t_start frames 

ms = pio.LoadMouse(path_npz)
cap = cv2.VideoCapture(path_beh)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.

out = cv2.VideoWriter(path_beh_out, cv2.VideoWriter_fourcc(*'MP4V'), 20, (frame_width,frame_height))


for t in range(t_start):
    ret, frame = cap.read()
  
for t in range(t_start, t_start+t_len):
    ret, frame = cap.read()
    for i,n in enumerate(nums):
        pf = mcl.FindPlaceField(ms, n)
        true_t = int(t - ms.time[0]*20)
        if true_t  < pf[0].t_spec:
            continue
        color = 255*(0.6 + 0.4*ms.neur[true_t,n]/max(ms.neur[:t_len,n]))*np.array(colornum_Metro(i+1))
        axes = (rad- (drad+gap)*lay[i], int((rad- (drad+gap)*lay[i])*4/3))
        cv2.ellipse(frame, center, axes, 0, 180+pf[0].mu-pf[0].std/2, 180+pf[0].mu+pf[0].std/2, color, drad)
    out.write(frame)

cap.release()
out.release()

#%% Merge grey neural video with a mask

cap = cv2.VideoCapture(path_neur)
mask = cv2.imread(path_mask)
greymask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
greymask = cv2.cvtColor(greymask, cv2.COLOR_GRAY2RGB)


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter(path_neur_out, cv2.VideoWriter_fourcc(*'MP4V'), 20, (frame_width,frame_height))        
while(cap.isOpened()):    
        ret, frrame = cap.read()
        if not ret:
            continue

        frrame[greymask>0] = mask[greymask>0]
        out.write(frrame)

cap.release()
out.release()  


#%% Merge grey neural video with each filter separately

cap = cv2.VideoCapture(path_neur)
mask = cv2.imread(path_mask)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter(path_neur_out, cv2.VideoWriter_fourcc(*'MP4V'), 20, (frame_width,frame_height))        
while(cap.isOpened()):    
        ret, frrame = cap.read()
        if not ret:
            continue
        for i, file in enumerate(os.listdir(path_filters)):
            filt = cv2.imread(path_filters+file)
            filt = (filt>0).astype('uint8')
            col_filt = filt.astype('float64').copy()
            for j,c in enumerate(drcl.colornum_Metro(i)):            
                col_filt[:,:,j] = filt[:,:,j].astype('float64')*c
                
            col_filt = col_filt/np.max(filt)
            frrame = frrame/np.max(frrame)
            col_filt[np.where(filt==0)] = 1
            frrame *= col_filt*255
        out.write(frrame.astype('uint8'))

cap.release()
out.release() 



#%%Merge two videos

cap1 = cv2.VideoCapture(path_neur_out)
cap2 = cv2.VideoCapture(path_beh_out)
out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'MP4V'), 20, (1197,513))

while(cap1.isOpened()):    
        ret, frame1 = cap1.read()  
        ret, frame2 = cap2.read()
        try:
            frame1 = frame1[3:,50:563]
            frame2 = cv2.resize(frame2[:,150:1230], (684,513))
            frame3 = cv2.hconcat([frame1, frame2])
            out.write(frame3)
        except:
            continue

cap1.release()
cap2.release()
out.release()  
