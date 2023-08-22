"""
Created on Fri aug 11 2023

@author: jca



"""
import time
import os
import cv2
import math
import numpy as np
import seaborn as sns
from numpy import empty
from scipy.stats import norm
from skimage.util import invert
import matplotlib.pyplot as plt
from skimage.morphology import label
from scipy.optimize import root_scalar
from skimage.morphology import skeletonize
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import remove_small_objects
from skimage.morphology import reconstruction
import Dictionary as dict
from scipy.stats import norm


start_time = time.time()
# # ================================================================================================================
# Inputs
# # ================================================================================================================
# Crack to check
n = 22
# Method, used to know where the final subimg is located and where files need to be saved
method_threshold = 3.5

# If the generated final marked image want to be saved
save_img = False
# If the generated reference point list want to be saved
saveref_points_list = False

# # ===============================================================================================================.
# 1. Paths arrangement
# # ===============================================================================================================.
# path to find the image
path1 = r'C:\Users\juanc\OneDrive - KTH\Journals\01-Quantification\Image_list\Crack ' + str(n) + '\\'
# Name of the folder where the final img is located

pathCracked= path1 +'00_Cracked_subimg\\'
pathunCracked= path1 +'01_Uncracked_subimg\\'
pathsubfolder = 'MAD k=' + str(method_threshold) + ' full_subimg''\\'
path2 = path1 + pathsubfolder

# Access the path
os.chdir(pathCracked)

# Get the subimage (selected image) and turns it into greyscale (imageBW)
# ========================================================================================================
selectedimage, imageBW = dict.selectimg('_DCS7271_164')

# Put the image pixels in order
WinData = imageBW.ravel()

# # ================================================================================================================
# What to show
# # ================================================================================================================

filters=True
postMAD=True
LoG=True
masks=True

# # ================================================================================================================
# Bilateral filter
# # ================================================================================================================
median = np.median(WinData)
mad = np.median(np.abs(WinData - median))
threshold=3.5
lower_bound = median - threshold * mad
distMAD_median=threshold * mad

bilateral = cv2.bilateralFilter(imageBW, d=9, sigmaColor=distMAD_median, sigmaSpace=10)

Databilateral= bilateral.ravel()

# # ================================================================================================================
# MAD and threshold
# # ================================================================================================================

Outlm5, medm5, medlistm5, inliersm5, trhsm5 = dict.detect_outliers_mad(Databilateral, method_threshold)
ret,bilateral_truncated = cv2.threshold(bilateral, trhsm5, 255, cv2.THRESH_TRUNC)

# # ================================================================================================================
# Erosion
# # ================================================================================================================

seed = np.copy(bilateral_truncated)
seed[1:-1, 1:-1] = bilateral_truncated.min()
mask = bilateral_truncated
bilateral_truncated_filled = reconstruction(seed, mask, method='dilation')


# # ================================================================================================================
#LoG
# # ================================================================================================================

# 2. bilateral
# ________________________________________________________________________________________________________
LoGbilateral = cv2.Laplacian(bilateral_truncated_filled, cv2.CV_64F)

# # ================================================================================================================
# Masks
# # ================================================================================================================

neg_mask_LoGbilateral = LoGbilateral<0
pos_mask_filled = bilateral_truncated_filled <np.floor(trhsm5)

width = LoGbilateral.shape[0]
height = LoGbilateral.shape[1]
finalsubimg = empty([width, height,3], dtype=np.uint8)
for x in range(0, width):
    for y in range(0, height):
        if pos_mask_filled[x, y] and LoGbilateral[x, y]<0:
            finalsubimg[x, y] = [255, 0, 0]  # If pixel is part of skeleton paint red
        elif pos_mask_filled[x, y]:
            finalsubimg[x, y] = [255, 255, 0]  # If pixel is part of crack paint yellow
        elif LoGbilateral[x, y]<0:
            finalsubimg[x, y] = [0, 0, 255]  # If pixel is part of edge, paint blue

        else:
            finalsubimg[x, y] = 0



if masks:
    plt.figure('Masks')
    plt.subplot(231)
    plt.title('bilateral')
    plt.imshow(bilateral, cmap='gray')
    plt.subplot(232)
    plt.title('bilateral_truncated')
    plt.imshow(bilateral_truncated, cmap='gray')
    plt.subplot(233)
    plt.title('bilateral_truncated_filled ')
    plt.imshow(bilateral_truncated_filled, cmap='gray')
    plt.subplot(234)
    plt.title('LoGbilateral')
    plt.imshow(LoGbilateral, cmap='gray')
    plt.subplot(235)
    plt.title('pos_mask_LoGbilateral ')
    plt.imshow(neg_mask_LoGbilateral, cmap='gray')
    plt.subplot(236)
    plt.title('pos_mask_filled')
    plt.imshow(pos_mask_filled, cmap='gray')


    plt.figure()
    plt.imshow(finalsubimg)
    plt.show(block=False)

# # ================================================================================================================
#
# # ================================================================================================================





# Finish time counter
end_time = time.time()
# Total time for the code
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")