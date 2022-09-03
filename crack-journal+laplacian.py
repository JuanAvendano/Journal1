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
from old_Crack_width_calculation import CrackWidth
from numpy import empty

# Load an image
# path = r'C:\Users\juanc\OneDrive - KTH\Python\Prueba\02.1-Cracked_predicted'
# os.chdir(path)  # Access the path
# image = cv2.imread('_DCS6695_412.jpg')
# image = cv2.imread('_DCS6695_412 - Copy.jpg')


path = r'C:\Users\juanc\Desktop\prueba'
os.chdir(path)  # Access the path
image = cv2.imread('_DCS6932_193.jpg')

pixel_width=0.1
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
winW = 28
winH = 28


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


# Histogram and CDF for the initial image
hist, bins = np.histogram(img.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

# Equalizaed image
cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')
img2 = cdf[img]
hist2, bins = np.histogram(img2.flatten(), 256, [0, 256])
cdf2 = hist2.cumsum()
cdf_normalized2 = cdf2 * hist2.max() / cdf2.max()

# plt.figure('img', figsize=(10, 10))
# plt.subplot(131)
# plt.imshow(img, cmap='gray')
# plt.title('Image')
#
# plt.subplot(132)
# # plt.plot(cdf_normalized, color='b')
# plt.hist(img.flatten(), 256, [0, 256], color='r')
# plt.xlim([0, 256])
# plt.legend(('cdf', 'histogram'), loc='upper left')
#
# # plt.subplot(2, 2, 3)
# # plt.imshow(img2, cmap='gray')
# # plt.title('Image2')
#
# plt.subplot(133)
# # plt.plot(cdf_normalized2, color='b')
# plt.hist(img2.flatten(), 256, [0, 256], color='r')
# plt.xlim([0, 256])
# plt.legend(('cdf2', 'histogram2'), loc='upper left')
# plt.show(block=False)

for (x, y, window) in sliding_window(img, stepSize=28, windowSize=(winW, winH)):
    # Histogram and balanced threshold for windows in the initial image
    red = np.histogram(window.ravel(), bins=256, range=[0, 256])
    clone = img.copy()
    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    # cv2.imshow("Window", clone)
    trhs = balanced_hist_thresholding(red)
    # threshold the image
    ret, thresh = cv2.threshold(window, trhs, 255, cv2.THRESH_BINARY)

    # Apply Laplacian operator in some higher datatype
    laplacian = cv2.Laplacian(window, cv2.CV_64F)
    # Another alternative, applying Gaussian Blur first (for example using a 3x3 box)
    gaussblur3 = cv2.GaussianBlur(window, (3, 3), 0)
    # And then applying Laplacian
    laplacian2 = cv2.Laplacian(gaussblur3, cv2.CV_64F)
    # #
    # laplacian1 = laplacian / laplacian.max()

    # Histogram and balanced threshold for windows after Laplacian
    red2 = np.histogram(laplacian2.ravel(), bins=512, range=[-256, 256])
    trhs20 = balanced_hist_thresholding(red2)
    trhs2 = red2[1][trhs20]
    ret2, thresh2 = cv2.threshold(laplacian, trhs2, 255, cv2.THRESH_BINARY)

    plt.figure('window hist trsh', figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(clone, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(window, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.hist(window.ravel(), 256, [0, 256])
    # plt.bar(red[1][:256], red[0])
    plt.subplot(2, 2, 4)
    plt.imshow(thresh, cmap='gray')

    plt.figure('laplacian', figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(clone, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(laplacian, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.hist(laplacian.ravel(), 256, [-256, 256])
    # plt.imshow(laplacian2, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(laplacian2, cmap='gray')

    plt.figure('balanced laplacian ', figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(clone, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(laplacian, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.hist(laplacian.ravel(), 512, [-256, 256])
    # plt.bar(red[1][:256], red[0])
    plt.subplot(2, 2, 4)
    plt.imshow(thresh2, cmap='gray')

    from skimage.measure import regionprops
    from skimage.measure import label as lb
    from skimage import measure


    """ 
    # threshreg = lb(thresh,connectivity=2)  # Two pixels are connected when they are neighbors and have the same value. In 2D, they can be neighbors either in a 1- or 2-connected sense.  The value refers to the maximum number of orthogonal hops to consider a  pixel/voxel a neighbor
    # prop=regionprops(threshreg)  # Acquiring regions properties

    # contours = measure.find_contours(threshreg, 0.5,'high')
    # # Display the image and plot all contours found
    # fig, ax = plt.subplots()
    # ax.imshow(threshreg, cmap=plt.cm.gray)
    #
    # for contour in contours:
    #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    #
    # ax.axis('image')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.show()

    """


    widths,coordsk,skframes,edgesframes = CrackWidth(thresh,pixel_width)

    width = skframes.shape[0]
    height = skframes.shape[1]
    # thrlap = cv2.Laplacian(thresh, cv2.CV_64F) #Laplacian of the thresholded image
    cracks = empty([width, height, 3], dtype=np.uint8)  # creates the image with the crack obtained.
    for n in range(0, width):
        for m in range(0, height):
            if skframes[n, m] > 0:          # If pixel is part of skeleton paint blue
                cracks[n, m] = [0, 0, 255]
            elif thresh[n, m] == 0:         # If pixel is part of crack paint yellow
                cracks[n, m] = [255, 255, 0]
                if edgesframes[n, m] == 255:  # If pixel is part of crack and part of edge, paint red
                # if thrlap[n, m] == 255: #<0
                    cracks[n, m] = [255, 0, 0]
            elif edgesframes[n, m] == 255:     # If pixel is part of edge, paint green
            # elif thrlap[n, m] == 255: #<0
                cracks[n, m] = [0, 255, 0]
            else:
                cracks[n, m] = 0
    plt.figure('skl+edges', figsize=(10, 10))
    plt.imshow(cracks, cmap='gray')


    if (len(widths) != 0 ):
        listcomplt = np.column_stack([coordsk, widths])
    # # Save those information into text files.

        f1 = open(path+"\\listcomplt"+str(x)+"_"+str(y)+".txt", 'w')
        for z in range(0, len(listcomplt)):
            f1.write(str(listcomplt[z]) + 'mm' + '\n')
        f1.close()
# ======================================================================================================================
# """CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
# # create a CLAHE object (Arguments are optional).
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# cl1 = clahe.apply(img)
#
# plt.figure('img clahe', figsize=(10, 10))
# plt.subplot(2, 2, 1)
# plt.imshow(img, cmap='gray')
# plt.title('Image')
#
# plt.subplot(2, 2, 2)
# plt.hist(img.flatten(), 256, [0, 256], color='r')
# plt.xlim([0, 256])
# plt.legend(('histogram'), loc='upper left')
#
# plt.subplot(2, 2, 3)
# plt.imshow(cl1, cmap='gray')
# plt.title('clahe')
#
# plt.subplot(2, 2, 4)
# plt.hist(cl1.flatten(), 256, [0, 256], color='r')
# plt.xlim([0, 256])
# plt.legend(('histogram2'), loc='upper left')
# plt.show()
# ======================================================================================================================
# region slide window


# endregion
