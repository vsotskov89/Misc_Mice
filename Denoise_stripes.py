# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 12:03:30 2021

@author: vsots
"""

import os.path
from os import path
import cv2
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm 
from scipy.signal import butter, lfilter, freqz, filtfilt

# Values users can modify:
dataDir = "D:\\Work\\Miniscope_test\\"
dataFilePrefix = ''
startingFileNum = 0
framesPerFile = 300
frameStep = 10 # Can use frame skipping to speed this up
showVideo = True
fileNum = startingFileNum
sumFFT = None
applyVignette = True
vignetteCreated = False
running = True
goodRadius = 2000
notchHalfWidth = 3
centerHalfHeightToLeave = 10
# -----------------------

rows = 600
cols = 600

# plt.rcParams["figure.figsize"] = (10,5)

print (dataDir + dataFilePrefix + "{:.0f}.avi".format(fileNum))

while (path.exists(dataDir + dataFilePrefix + "{:.0f}.avi".format(fileNum)) and running is True):
    cap = cv2.VideoCapture(dataDir + dataFilePrefix + "{:.0f}.avi".format(fileNum))
    fileNum = fileNum + 1
    frameNum = 0
    for frameNum in tqdm(range(0,framesPerFile, frameStep), total = framesPerFile/frameStep, desc ="Running file {:.0f}.avi".format(fileNum - 1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        ret, frame = cap.read()

        if (vignetteCreated is False):
            rows, cols = frame.shape[:2] 
            X_resultant_kernel = cv2.getGaussianKernel(cols,cols/4) 
            Y_resultant_kernel = cv2.getGaussianKernel(rows,rows/4) 
            resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T 
            mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
            vignetteCreated = True

        if applyVignette is False:
            mask = mask * 0 + 1
        
        if (ret is False):
            break
        else:
            frame = frame[:,:,1] * mask
            
            dft = cv2.dft(np.float32(frame),flags = cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
             
            try:
                sumFFT = sumFFT + cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])
            except:
                sumFFT = cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])

            # if (showVideo is True):
            #     cv2.imshow("Vid", frame/255)
            #     if cv2.waitKey(10) & 0xFF == ord('q'):
            #         running = False
            #         break

crow,ccol = int(rows/2) , int(cols/2)

maskFFT = np.zeros((rows,cols,2), np.float32)
cv2.circle(maskFFT,(crow,ccol),goodRadius,1,thickness=-1)

# for i in cutFreq:
#     maskFFT[(i + crow-notchHalfWidth):(i+crow+notchHalfWidth),(ccol-notchHalfWidth):(ccol+notchHalfWidth),0] = 0
#     maskFFT[(-i + crow-notchHalfWidth):(-i+crow+notchHalfWidth),(ccol-notchHalfWidth):(ccol+notchHalfWidth),0] = 0
maskFFT[(crow+centerHalfHeightToLeave):,(ccol-notchHalfWidth):(ccol+notchHalfWidth),0] = 0
maskFFT[:(crow-centerHalfHeightToLeave),(ccol-notchHalfWidth):(ccol+notchHalfWidth),0] = 0

maskFFT[:,:,1] = maskFFT[:,:,0]


modifiedFFT = sumFFT * maskFFT[:,:,0]

# Plot original and modified FFT
# plt.figure()
# plt.subplot(121),plt.imshow(np.log(sumFFT), cmap = 'gray')
# plt.title('Mean FFT of Data')
# plt.subplot(122),plt.imshow(np.log(modifiedFFT), cmap = 'gray')
# plt.title('Filtered FFT')

# Display filtered vs original videos

# Values users can modify:
frameStep = 3 # This will speed up the playback
# -----------------------

fileNum = startingFileNum
sumFFT = None
running = True

while (path.exists(dataDir + dataFilePrefix + "{:.0f}.avi".format(fileNum)) and running is True):
    cap = cv2.VideoCapture(dataDir + dataFilePrefix + "{:.0f}.avi".format(fileNum))
    fileNum = fileNum + 1
    for frameNum in tqdm(range(0,framesPerFile, frameStep), total = framesPerFile/frameStep, desc ="Running file {:.0f}.avi".format(fileNum - 1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        ret, frame = cap.read()
        
        if (ret is False):
            break
        else:
            frame = frame[:,:,1]
            dft = cv2.dft(np.float32(frame),flags = cv2.DFT_COMPLEX_OUTPUT|cv2.DFT_SCALE)
            dft_shift = np.fft.fftshift(dft)
             
            fshift = dft_shift * maskFFT
            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
            
            img_back[img_back >255] = 255
            img_back = np.uint8(img_back)

            im_diff = (128 + (frame - img_back)*2)
            im_v = cv2.hconcat([frame, img_back, im_diff])
            cv2.imshow("Raw, Filtered, Difference", im_v/255)

            try:
                sumFFT = sumFFT + cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])
            except:
                sumFFT = cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])

            if cv2.waitKey(10) & 0xFF == ord('q'):
                running = False
                break

#cv2.destroyAllWindows()

# Calculate mean fluorescence per frame

# Users shouldn't change anything here
frameStep = 1 # Should stay as 1
fileNum = startingFileNum
sumFFT = None
meanFrameList = []


while (path.exists(dataDir + dataFilePrefix + "{:.0f}.avi".format(fileNum))):
    cap = cv2.VideoCapture(dataDir + dataFilePrefix + "{:.0f}.avi".format(fileNum))
    fileNum = fileNum + 1
    
    if fileNum >= 7:
        break
        
    for frameNum in tqdm(range(0,framesPerFile, frameStep), total = framesPerFile/frameStep, desc ="Running file {:.0f}.avi".format(fileNum - 1)):
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        ret, frame = cap.read()
        
        if (ret is False):
            break
        else:
            frame = frame[:,:,1]
            dft = cv2.dft(np.float32(frame),flags = cv2.DFT_COMPLEX_OUTPUT|cv2.DFT_SCALE)
            dft_shift = np.fft.fftshift(dft)
             
            fshift = dft_shift * maskFFT
            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
            meanFrameList.append(img_back.mean())
            
            # clear_output(wait=True)

            # plt.subplot(121),plt.imshow(frame, cmap = 'gray')
            # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
            # plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
            # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

            # plt.show()

meanFrame = np.array(meanFrameList)


# Create a lowpass filter
# Sample rate and desired cutoff frequencies (in Hz).

# Values users can modify:
fs = 20 # TODO: Should get this from timestamp file
cutoff = 2.0
# -----------------------

# plt.figure()
# for order in [3, 6, 9]:
#     b, a = butter(order, cutoff/ (0.5 * fs), btype='low', analog = False)
#     w, h = freqz(b, a, worN=2000)
#     plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

# plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
#              '--', label='sqrt(0.5)')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Gain')
# plt.grid(True)
# plt.legend(loc='best')
# # Plot Mean Frame Resuls
# # plt.figure(figsize=(8,4))
# # plt.plot(meanFrame)

# Plot effect of filtering

# Values users can modify:
butterOrder = 3
# -----------------------

b, a = butter(butterOrder, cutoff/ (0.5 * fs), btype='low', analog = False)
meanFiltered = filtfilt(b,a,meanFrame)
# plt.figure()
# plt.plot(meanFrame, 'k', label='Raw Data')
# plt.plot( meanFiltered, label='Filtered Data')
# plt.plot(meanFrame - meanFiltered,'r', label='Difference')
# plt.xlabel('frame number')
# # plt.hlines([-a, a], 0, T, linestyles='--')
# plt.grid(True)
# plt.axis('tight')
# plt.legend(loc='upper left')
# meanFrame[3000]

# Apply FFT spatial filtering and lowpass filtering to data and has the option of saving as new videos

# Values users can modify:
# Select one below -
# mode = "display"
mode = 'save'

frameStep = 1 #1 # Should be set to 1 for saving

# Select one below -
compressionCodec = "FFV1"
# compressionCodec = "GREY"
# --------------------

fileNum = startingFileNum
sumFFT = None
frameCount = 0
running = True

if (mode is "save" and frameStep is not 1):
    print("WARNING: You are only saving every {} frame!".format(frameStep))

codec = cv2.VideoWriter_fourcc(compressionCodec[0],compressionCodec[1],compressionCodec[2],compressionCodec[3])

if (mode is "save" and not path.exists(dataDir + "Denoised")):
    os.mkdir(dataDir + "Denoised")

while (path.exists(dataDir + dataFilePrefix + "{:.0f}.avi".format(fileNum)) and running is True):
    cap = cv2.VideoCapture(dataDir + dataFilePrefix + "{:.0f}.avi".format(fileNum))

    if (mode is "save"):
        writeFile = cv2.VideoWriter(dataDir + "Denoised/" + dataFilePrefix + "denoised{:.0f}.avi".format(fileNum),  
                            codec, 60, (cols,rows), isColor=False) 

    fileNum = fileNum + 1
    # frameNum = 0
    for frameNum in tqdm(range(0,framesPerFile, frameStep), total = framesPerFile/frameStep, desc ="Running file {:.0f}.avi".format(fileNum - 1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        ret, frame = cap.read()
        # frameNum = frameNum + frameStep 
        
        # print(frameCount)
        
        if (ret is False):
            break
        else:
            frame = frame[:,:,1]
            dft = cv2.dft(np.float32(frame),flags = cv2.DFT_COMPLEX_OUTPUT|cv2.DFT_SCALE)
            dft_shift = np.fft.fftshift(dft)
             
            fshift = dft_shift * maskFFT
            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

            meanF = img_back.mean()
            img_back = img_back * (1 + (meanFiltered[frameCount] - meanF)/meanF)
            img_back[img_back >255] = 255
            img_back = np.uint8(img_back)

            
            
            if (mode is "save"):
                writeFile.write(img_back)

            # if (mode is "display"):
            #     im_diff = (128 + (frame - img_back)*2)
            #     im_v = cv2.hconcat([frame, img_back])
            #     im_v = cv2.hconcat([im_v, im_diff])

            #     im_v = cv2.hconcat([frame, img_back, im_diff])
            #     cv2.imshow("Cleaned video", im_v/255)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         running = False
            #         cap.release()
            #         break

            frameCount = frameCount + 1

    if (mode is "save"):
        writeFile.release()

# cv2.destroyAllWindows()



