import cv2
from skimage.morphology import skeletonize
import numpy as np
import matplotlib.pyplot as plt
import os
import Dictionary as dict
from numpy import empty
import time
import glob

path = r'C:\Users\juanc\OneDrive - KTH\Journals\01-Quantification\Image_list\Crack 6\MAD k=2'

os.chdir(path)  # Access the path
img = cv2.imread('res_DCS7230_215.png')
image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgbin=(image //255)
# perform skeletonization
skeleton = skeletonize(imgbin)

# perform edge detection
imim = (imgbin == 1)
imim.dtype = 'uint8'

edges = cv2.Laplacian(imgbin, cv2.CV_64F)

# threshold edges to binary image
thrlap = np.zeros_like(edges)
thrlap[edges > 0] = 255

# get skeleton coordinates
skeleton_coords = np.argwhere(skeleton > 0)

sumimg=thrlap.copy()
for i, (row, col) in enumerate(skeleton_coords):
    sumimg[row,col]=255

# initialize width array
width = np.zeros(len(skeleton_coords))

# loop over skeleton pixels
for i, (row, col) in enumerate(skeleton_coords):

    # calculate angle of skeleton at pixel
    grad_x, grad_y = np.gradient(skeleton.astype(float))
    grad_mag = np.sqrt(grad_x[row, col] ** 2 + grad_y[row, col] ** 2)
    if grad_mag > 0:
        angle = np.arctan2(grad_y[row, col], grad_x[row, col])
    else:
        angle = np.pi / 2

    # calculate perpendicular direction
    perp_angle1 = angle + np.pi / 2
    perp_angle2 = angle - np.pi / 2

    # calculate unit vectors in perpendicular directions
    perp_dir1 = np.array([np.cos(perp_angle1), np.sin(perp_angle1)])
    perp_dir2 = np.array([np.cos(perp_angle2), np.sin(perp_angle2)])
    perp_dir1 /= np.linalg.norm(perp_dir1) + 1e-8
    perp_dir2 /= np.linalg.norm(perp_dir2) + 1e-8

    # check pixels in perpendicular directions
    pixel_found1 = False
    pixel_found2 = False
    pixel_count = 0
    while not (pixel_found1 and pixel_found2) and pixel_count < 20:
        pixel_count += 1
        offset1 = pixel_count * perp_dir1
        offset2 = pixel_count * perp_dir2
        new_row1 = int(round(row + offset1[0]))
        new_col1 = int(round(col + offset1[1]))
        new_row2 = int(round(row + offset2[0]))
        new_col2 = int(round(col + offset2[1]))

        # check if new pixels are outside image bounds
        if (new_row1 < 0 or new_row1 >= image.shape[0] or new_col1 < 0 or new_col1 >= image.shape[1]) \
                and (new_row2 < 0 or new_row2 >= image.shape[0] or new_col2 < 0 or new_col2 >= image.shape[1]):
            continue

        # check if new pixels are part of edge
        if not pixel_found1 and thrlap[new_row1, new_col1] > 0:
            pixel_found1 = True
            width[i] += np.linalg.norm(offset1)
        if not pixel_found2 and thrlap[new_row2, new_col2] > 0:
            pixel_found2 = True
            width[i] += np.linalg.norm(offset2)

# add width values to results array
results = np.hstack((np.array(skeleton_coords), width.reshape(-1, 1)))


sumres=sumimg.copy()
for i, (row, col) in enumerate(skeleton_coords):
    sumres[row,col]=results[i][2]+150
print(width)