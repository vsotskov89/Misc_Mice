# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:01:43 2020

@author: vsots
"""
import serial
import ctypes
import ctypes.wintypes
import sys


baud_rate = 9600 
com_port = 'COM' 
fout_path = 'D:\Work\AMYG_PTSD\AP_03_ST_sync.txt'

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
current_time = ctypes.wintypes.LARGE_INTEGER()

for i in range(9):  #searching for a proper port digit
    try:
        port = serial.Serial(com_port + str(i), baud_rate)
        break
    except:
        continue
    

fout = open(fout_path, 'w')
fout.write('time_s\tvoltage_v\n')
i = 0

while port.read(size=1) != b'\n':
    pass

while 1:
    try:
        kernel32.QueryPerformanceCounter(ctypes.byref(current_time))
        msg = port.read(size=6)
        i += 1
        if not (i % 4):
            fout.write(str(current_time.value/10000000) + '\t' + str(msg.decode()[0:-1]))
    except KeyboardInterrupt:
        fout.close()
        port.close()
        print('Closed')
        break
    

