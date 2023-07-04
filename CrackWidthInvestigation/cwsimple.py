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
results=np.zeros([len(skeleton_coords),3])
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
    perp_angle = angle + np.pi / 2

    # calculate unit vector in perpendicular direction
    perp_dir = np.array([np.cos(perp_angle), np.sin(perp_angle)])
    perp_dir /= np.linalg.norm(perp_dir) + 1e-8

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
            width[i] = np.linalg.norm(offset)
    results[i][0:3]=col,row,width[i]

sumres=sumimg.copy()
for i, (row, col) in enumerate(skeleton_coords):
    sumres[row,col]=results[i][2]+150
print(width)