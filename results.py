"""
Created on Wen march 24 2022

@author: jca
sliding window: https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
balanced histogram: https://theailearner.com/2019/07/19/balanced-histogram-thresholding/

beta version of the crack journal+laplacian to see if by changing it, it is still ok
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import face
from PIL import Image, ImageDraw
import os
import cv2
# from Crack_width_calculation import CrackWidth
from numpy import empty
import Dictionary as dict
from skimage.morphology import label
from skimage.morphology import remove_small_objects
from skimage.util import invert




# Load an image
# path = r'C:\Users\juanc\OneDrive - KTH\Python\Prueba\02.1-Cracked_predicted'
# os.chdir(path)  # Access the path
# image = cv2.imread('_DCS6695_412.jpg')
# image = cv2.imread('_DCS6695_412 - Copy.jpg')


path = r'C:\Users\juanc\OneDrive - KTH\Journals\01-Quantification\Image_list\Crack 1'
os.chdir(path)  # Access the path

pixel_width=0.08
winW = 28
winH = 28
crack=0
# Crack 1
_DCS6931_078= [[168, 140], [168, 168], [168, 196], [196, 196]]
_DCS6931_111= [[168, 0], [196, 0], [168, 28], [168, 56], [168, 84], [168, 112], [168, 140], [140,140], [140, 168], [112, 168],  [112, 196], [140, 196]]
_DCS6931_144= [[112, 0], [140, 0], [140, 28], [140, 56], [112, 84], [112, 112], [84, 140], [112, 140], [84, 168], [112, 168], [84, 196], [112, 196]]
_DCS6931_177= [[84, 0], [112, 0], [84, 28], [84, 56], [84, 84], [84, 112], [84, 140], [84, 168], [84, 196]]
_DCS6931_210= [[84, 0], [84, 28], [56, 56], [56, 140], [84, 140], [84, 168], [84, 196], [112, 196]]
_DCS6931_243= [[112, 0], [112, 28], [112, 56], [112, 84], [112, 112], [112, 140], [112, 168], [112, 196]]
_DCS6931_276= [[196, 0]]

crack1=['_DCS6931_078','_DCS6931_111','_DCS6931_144','_DCS6931_177','_DCS6931_210','_DCS6931_243','_DCS6931_276']
WindowsCrack1=[_DCS6931_078, _DCS6931_111, _DCS6931_144, _DCS6931_177, _DCS6931_210, _DCS6931_243, _DCS6931_276]

crack=crack1
windows=WindowsCrack1
resultlis=[]

def sliding_window(image, stepSize, windowSize):
    """
    Sliding window within the loaded image. The window corresponds to a squared window that slides across the image
    according to the step size.

     Parameters
    ----------
    image : ndarray
            Image where the window will slide.
    StepSize :  Stride for the window
    windowSize : Size of the squared window

    """


    # slide a window across the image
    for y in range(56, image.shape[0], stepSize):
        for x in range(28, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def balanced_hist_thresholding(b):
    # Starting point of histogram
    i_s = np.min(np.where(b[0] > 0))
    # End point of histogram
    i_e = np.max(np.where(b[0] > 0))
    # Center of histogram
    i_m = (i_s + i_e) // 2
    # Left side weight
    w_l = np.sum(b[0][0:i_m + 1])
    # Right side weight
    w_r = np.sum(b[0][i_m + 1:i_e + 1])
    # Until starting point not equal to endpoint
    while (i_s != i_e):
        # If right side is heavier
        if (w_r > w_l):
            # Remove the end weight
            w_r -= b[0][i_e]
            i_e -= 1
            # Adjust the center position and recompute the weights
            if ((i_s + i_e) // 2) < i_m:
                w_l -= b[0][i_m]
                w_r += b[0][i_m]
                i_m -= 1
        else:
            # If left side is heavier, remove the starting weight
            w_l -= b[0][i_s]
            i_s += 1
            # Adjust the center position and recompute the weights
            if ((i_s + i_e) // 2) >= i_m:
                w_l += b[0][i_m + 1]
                w_r -= b[0][i_m + 1]
                i_m += 1
    return i_m


def selectimg(crack,i):
        image = cv2.imread(crack[i]+'.jpg')
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image,img

def joinwindows(img,windows,i):
    resultImg = img.copy() * 0
    clone = img.copy()

    for j in range(0, len(windows[i])):
        x=windows[i][j][0]                      # x coordinate of the upper left corner of the window to evaluate
        y=windows[i][j][1]                      # y coordinate of the upper left corner of the window to evaluate
        window = img[y:y + winH, x:x + winW]  # window to evaluate from x and y coord
        red = np.histogram(window.ravel(), bins=256, range=[0, 256])   # hist for the window
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)    # rectangle to see where we are in the image

        trhs = balanced_hist_thresholding(red)
    # ====================================================================================================================================================================================================
        if i==1 and j==6:
            trhs=55
    # ====================================================================================================================================================================================================
        ret, thresh = cv2.threshold(window, trhs, 255, cv2.THRESH_BINARY)
        thresh=invert(thresh)
        xx = 0
        yy = 0
        for k in range(x, x+window.shape[1]):

            for l in range(y, y+window.shape[0]):
                resultImg[l,k]=thresh[xx,yy]
                xx+=1
            yy += 1
            xx=0
        plt.figure('window hist trsh', figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(img)
        plt.subplot(2, 2, 2)
        plt.imshow(window, cmap='gray')
        plt.subplot(2, 2, 3)
        # plt.hist(window.ravel(), 256, [0, 256])
        plt.imshow(clone)
        plt.subplot(2, 2, 4)
        plt.imshow(resultImg, cmap='gray')



    return resultImg,window

for i in range (0, len(crack)):
    primera,primeraBW=selectimg(crack,i)
    resultado,wind=joinwindows(primeraBW,windows,i)
    resultlis.append(resultado)
    # ====================================================================================================================================================================================================
    """ ESP: Probablemente esta parte se puede hacer como una funcion, por el momento la dejo asi para poder ir viendo
    que da como resultado regions o no_smllobj en caso de necesitarlo"""
    regions0=label(resultado,connectivity=2)
    no_smllobj = remove_small_objects(regions0, min_size=3)  # remove objects with dimensions smaller than 3 pixels
    regions = label(no_smllobj, connectivity=2) # creates regions again after having small objects removed
    resultImage = (regions > 0.5) * 255  # We create the binary image
    # ====================================================================================================================================================================================================
    resname='res2'+crack[i] # name of the image that will be saved
    dict.imgSaving(path, resname, resultImage) #the image where small object have been removed is saved in the path

    # widths: list of of widths per pixel of skeleton
    # coordsk: List of coordinates of the skeleton's pixels
    # skframes: Skeleton image
    # edgesframes: Image of the edges of the crack
    # completeList: List of results, creates a vector with x,y coord of the skeleton and the corresponding width for
    #               that skeleton pixel and in mm according to the measure of pixel width
    widths,coordsk,skframes,edgesframes,completeList = dict.CrackWidth(resultImage//255,pixel_width)
    skframesname = 'skframes_' + crack[i]  # name of the skeleton image that will be saved
    edgesframesname = 'edgesframes_' + crack[i]  # name of the edges image that will be saved
    completeListname = 'completeList_' + crack[i]+'.txt'  # name of the image that will be saved
    dict.imgSaving(path, skframesname, skframes)  # the image where skeleton is saved in the path
    dict.imgSaving(path, edgesframesname, edgesframes)  # the image where edges of the crack is saved in the path
    with open(path + '//'+completeListname, "w") as output:
        output.write(str(completeList))


    plt.figure('window hist trsh', figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(primera)
    plt.subplot(2, 2, 2)
    plt.imshow(primeraBW, cmap='gray')
    plt.subplot(2, 2, 3)
    # plt.hist(window.ravel(), 256, [0, 256])
    plt.imshow(wind)
    plt.subplot(2, 2, 4)
    plt.imshow(resultado, cmap='gray')
    plt.show()

