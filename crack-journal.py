"""
Created on Wen march 24 2022

@author: jca
sliding window: https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
balanced histogram: https://theailearner.com/2019/07/19/balanced-histogram-thresholding/

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import face
from PIL import Image, ImageDraw
import os
import cv2

# Load an image
path = r'C:\Users\juanc\OneDrive - KTH\Python\Prueba\02.1-Cracked_predicted'
os.chdir(path)  # Access the path
image = cv2.imread('_DCS6695_412.jpg')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
winW=28
winH=28

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(84, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def balanced_hist_thresholding(b):
    # Starting point of histogram
    i_s = np.min(np.where(b[0]>0))
    # End point of histogram
    i_e = np.max(np.where(b[0]>0))
    # Center of histogram
    i_m = (i_s + i_e)//2
    # Left side weight
    w_l = np.sum(b[0][0:i_m+1])
    # Right side weight
    w_r = np.sum(b[0][i_m+1:i_e+1])
    # Until starting point not equal to endpoint
    while (i_s != i_e):
        # If right side is heavier
        if (w_r > w_l):
            # Remove the end weight
            w_r -= b[0][i_e]
            i_e -= 1
            # Adjust the center position and recompute the weights
            if ((i_s+i_e)//2) < i_m:
                w_l -= b[0][i_m]
                w_r += b[0][i_m]
                i_m -= 1
        else:
            # If left side is heavier, remove the starting weight
            w_l -= b[0][i_s]
            i_s += 1
            # Adjust the center position and recompute the weights
            if ((i_s+i_e)//2) >= i_m:
                w_l += b[0][i_m+1]
                w_r -= b[0][i_m+1]
                i_m += 1
    return i_m






# Histogram and CDF for the initial image
hist, bins = np.histogram(img.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()




# Equalizaed image
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img2 = cdf[img]
hist2, bins = np.histogram(img2.flatten(), 256, [0, 256])
cdf2 = hist2.cumsum()
cdf_normalized2 = cdf2 * hist2.max() / cdf2.max()



plt.figure('img',figsize=(10, 10))
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Image')

plt.subplot(132)
# plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')

# plt.subplot(2, 2, 3)
# plt.imshow(img2, cmap='gray')
# plt.title('Image2')

plt.subplot(133)
# plt.plot(cdf_normalized2, color='b')
plt.hist(img2.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf2', 'histogram2'), loc='upper left')
plt.show()

for (x, y, window) in sliding_window(img, stepSize=28, windowSize=(winW, winH)):
    red = np.histogram(window[..., 0].ravel(), bins=256, range=[0, 256])
    clone = img.copy()
    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    # cv2.imshow("Window", clone)
    trhs=balanced_hist_thresholding(red)
    plt.figure('imggdfgw', figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(clone, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(window, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.hist(window.ravel(), 256, [0, 256])
    # plt.bar(red[1][:256], red[0])
    # threshold the image
    ret, thresh = cv2.threshold(window, trhs, 255, cv2.THRESH_BINARY)
    plt.subplot(2, 2, 4)
    plt.imshow(thresh, cmap='gray')

# ======================================================================================================================
"""CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

plt.figure('img clahe',figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Image')

plt.subplot(2, 2, 2)
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('histogram'), loc='upper left')

plt.subplot(2, 2, 3)
plt.imshow(cl1, cmap='gray')
plt.title('clahe')

plt.subplot(2, 2, 4)
plt.hist(cl1.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('histogram2'), loc='upper left')
plt.show()
# ======================================================================================================================
# region slide window


#endregion