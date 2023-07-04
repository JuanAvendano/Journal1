import cv2
from skimage.morphology import skeletonize
import numpy as np
import matplotlib.pyplot as plt
import os
import Dictionary as dict
from numpy import empty
import time
import glob

def crack_width(image):
    # perform skeletonization
    skeleton = skeletonize(image > 0)

    # perform edge detection
    imim = (image == 1)
    imim.dtype = 'uint8'

    edges = cv2.Laplacian(imim, cv2.CV_64F)

    # threshold edges to binary image
    thrlap = np.zeros_like(edges)
    thrlap[edges > 0] = 255

    # get skeleton coordinates
    skeleton_coords = np.argwhere(skeleton > 0)

    sumimg=thrlap
    for i, (row, col) in enumerate(skeleton_coords):
        sumimg[row,col]=255

    # initialize width array
    width = np.zeros(len(skeleton_coords))

    # loop over skeleton pixels
    for i, (row, col) in enumerate(skeleton_coords):
        # create circle around skeleton pixel
        circle_row, circle_col = np.ogrid[-2:3, -2:3]
        circle = np.sqrt(circle_row ** 2 + circle_col ** 2) <= 2

        # calculate angle of skeleton at pixel
        grad_x, grad_y = np.gradient(skeleton.astype(float))
        angle = np.arctan2(grad_y[row, col], grad_x[row, col])

        # calculate perpendicular direction
        perp_angle = angle + np.pi / 2

        # calculate unit vector in perpendicular direction
        perp_dir = np.array([np.cos(perp_angle), np.sin(perp_angle)])

        # check pixels in perpendicular direction
        pixel_found = False
        pixel_count = 0
        while not pixel_found and pixel_count < 20:
            pixel_count += 1
            offset = pixel_count * perp_dir
            new_row = int(round(row + offset[0]))
            new_col = int(round(col + offset[1]))

            # check if new pixel is outside image bounds
            if new_row < 0 or new_row >= image.shape[0] or new_col < 0 or new_col >= image.shape[1]:
                continue

            # check if new pixel is part of edge
            if thrlap[new_row, new_col] > 0:
                pixel_found = True
                width[i] = np.linalg.norm(offset) - 0.5

    return width


# Path for the different possible cracks
path = r'C:\Users\juanc\OneDrive - KTH\Journals\01-Quantification\Image_list'
# Number of the cracked that is going to be processed
n = 6
# Must be 0 if method is Balanced histogram. If it is MAD the value is the threshold value
method_threshold = 2
# pixel_width in mm
pixel_width = 0.08
# If the info related to x,y coordinates and widths want to be saved as text file
save_info = False
# If image without small object, skeletons, edges want to be saved as png
save_img_parts = False
# If the generated final subimages want to be saved
save_subimg = False
# If the generated images want to be saved
save_img = False
# Size of the window to use on every subimage
winAarray = [28]  # , 32, 37, 56


# region crack info
# Initialize list crack
crack = 0
# Crack 6
_DCS7230_183 = [[0, 196], [112, 196], [140, 196], [168, 196], [196, 196], [196, 196]]
_DCS7230_184 = [[112, 112], [140, 112], [168, 112], [196, 112], [0, 140], [56, 140], [84, 140], [0, 168], [56, 168]]
_DCS7230_185 = [[56, 84], [84, 84], [28, 112], [84, 112], [112, 112], [140, 112], [168, 112], [196, 140], [0, 168]]
_DCS7230_186 = [[140, 112], [168, 112], [28, 140], [56, 140], [84, 140], [112, 140], [140, 140]]
_DCS7230_215 = [[84, 0], [28, 28], [56, 28], [84, 28], [112, 28], [140, 28], [168, 28], [196, 28], [0, 56]]
_DCS7230_216 = [[56, 0], [84, 0], [112, 0], [28, 28], [56, 28]]

Crack6 = [ '_DCS7230_186']

WindowsCrack6 = [ _DCS7230_186]


# Crack dimensions in terms of rows and columns of subimages for each crack (first element is the number of the crack,
# second number is the number of columns and third is rows)

crackgeometry = [[1, 1, 7], [2, 5, 2], [3, 2, 8], [4, 9, 3], [5, 5, 3], [6, 5, 3], [7, 4, 6], [8, 5, 2], [9, 5, 4],
                 [10, 6, 3], [11, 6, 8], [12, 6, 2], [13, 6, 2]]

# endregion

# list of cracks to check
crack = [Crack6]
# list of windows for each subimage
windows = [ WindowsCrack6]

# List for the subimages results
resultlis = []
for h in range(0, len(crack)):
    # # ================================================================================================================
    # 1. Paths arrangement
    # # ================================================================================================================
    # Name of the folder where the information of the desired crack is located
    pathsubfolder = '\Crack ' + str(h + n)
    path2 = path + pathsubfolder  # Complete the path name with the folder name

    # Path where results will be saved if using MAD
    pathMAD = path2 + '\MAD k=' + str(method_threshold)
    # Path where results will be saved if using Balanced Thresholding
    pathBHist = path2 + '\Balanced'
    # Selection of the path to be used according to the method selected
    if method_threshold == 0:
        path3 = pathBHist
    else:
        path3 = pathMAD

    # # ================================================================================================================
    # 2. Work over the crack used.
    # # ================================================================================================================
    for i in range(0, len(crack[h])):
        # Access the path
        os.chdir(path2)

        # # ===========================================================================================================
        # 2.1 Process over each subimage for the different windows sizes.
        # # ===========================================================================================================
        for k in range(0, len(winAarray)):
            # Size of the window used (must be squared)
            winH = winAarray[k]
            winW = winH

            # 2.1.1 Selects image
            # Get the subimage (selected image) and turns it into greyscale (imageBW)
            # ========================================================================================================
            selectedimage, imageBW = dict.selectimg(crack[h][i])

            # 2.1.2 Method and joining
            # Applies method and joins the windows for each subimage together to create a subimage with only the crack
            # =======================================================================================================
            resultado, wind = dict.joinwindows(imageBW, windows[h], i, winH, winW, method_threshold)

            # Adds the result to a list
            resultlis.append(resultado)

            # 2.1.3 Remove small objects
            # Takes the image with only the crack and removes small objects according to the specified size
            # ========================================================================================================
            resultImage = dict.cleanimage(resultado, 3)

            width=crack_width(resultImage//255)