"""
Created on Fri may 16 2023

@author: jca

Creates  MAD result on full subimg, it does not consider the windows that are listed in unit result for example
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import Dictionary as dict
import seaborn as sns
from scipy.stats import norm
from skimage.util import invert
import glob
import xlsxwriter
# from Crack_width_calculation import CrackWidth
from numpy import empty
from scipy.misc import face
from PIL import Image, ImageDraw
import time


start_time = time.time()

def MADfull_subimg(n, k, pix_width, save_finalsubimg,save_final_full_subimage):
    # # ================================================================================================================
    # Inputs
    # # ================================================================================================================
    # Crack to check
    n=n
    # Method, used to know where the final subimg is located and where files need to be saved
    method_threshold = k
    # pixel width
    pixel_width=pix_width
    # If the file including the reference measurements, method measurements and errors wants to be saved
    save_finalsubimg=save_finalsubimg

    save_final_full_subimage=save_final_full_subimage

    crackgeometry = [[1, 1, 7], [2, 5, 2], [3, 2, 8], [4, 9, 3], [5, 5, 3], [6, 5, 3], [7, 4, 6], [8, 5, 2], [9, 5, 4],
                     [10, 6, 3], [11, 6, 8], [12, 6, 2], [13, 6, 2],[14,8,10],[15,7,8],[16,4,18],[17,3,9],[18,4,13],[19,3,10],[20,3,7],[21,5,11],[22,6,9],[23,4,13],[24,5,17],[25,6,16],[26,4,10],[27,6,4],[28,17,3],[29,15,4]
]
    # # ===============================================================================================================.
    # 1. Paths arrangement
    # # ===============================================================================================================.
    # path for crack
    path =r'C:\Users\jcac\OneDrive - KTH\Journals\01-Quantification\Image_list\Crack ' +str(n)+'\\'
    # Name of the folder where the cracked subimg are located
    pathsubfolder = '00_Cracked_subimg'
    path2 = path+pathsubfolder
    try:
        os.chdir(path)  # Access the path
        # Create the full_img folder if it doesn't exist
        os.mkdir(os.path.join(path, 'MAD k=' + str(method_threshold) + ' full_subimg'))
        print(f'Folder created successfully in {path2}')
    except OSError as e:
        print(f'Error creating folder in {path2}: {str(e)}')

    path3 = path + '\MAD k=' + str(method_threshold) + ' full_subimg'


    # # ================================================================================================================
    # 2. Process
    # # ================================================================================================================
    # List of paths where the cracked images are
    img_list = glob.glob(path2 + '\*.png')
    # Load the cracked images as a list
    #TODO: change the use of PIL
    images = [Image.open(path) for path in img_list]
    for i in range(0, len(images)):
        # Convert the image to a numpy array
        images[i] = np.array(images[i])

    cBW = images.copy()
    for i in range(0, len(images)):
        cBW[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)  # Cracks in grayscale

    for i in range(0,len(cBW)):
        # MAD, clean image, CrackWidth to generate final subimage, GrayScale image with crack highlighted
        resultImg, finalsubimg, low_bound, medlist, Outl=dict.final_subimage(cBW[i],cBW[i],method_threshold,pixel_width)

        # Save the resulting processed crack
        if save_finalsubimg == True:
            finalsubimgname =path3.split('\\')[9]+img_list[i].split('\\')[9].split('.')[0]  # Name for the resulting processed crack
            dict.imgSaving(path3, finalsubimgname, finalsubimg)  # Save image in the corresponding path

    # # ================================================================================================================
    # 3. Image with the crack obtained joining the different subimages processed
    # # ================================================================================================================
    # 3.1 Image geometry in terms of subimages
    # # ===============================================================================================================

    # Columns of subimages for the final image
    nc = crackgeometry[n-1][1]
    # Rows of subimages for the final image
    nr = crackgeometry[n-1][2]

    # 3.2 List of paths for the cracked processed and uncracked subimages
    # # ===============================================================================================================
    # List of cracked processed subimages (finalsubimg) paths
    subimglist = [os.path.join(path3, f) for f in os.listdir(path3) if "full_subimg" in f and ".png" in f]
    # Path where the uncracked sub images are for the current crack
    path4 = path + '01_Uncracked_subimg\\'
    # List of uncracked subimages paths
    uncrksubimglist = glob.glob(path4 + '*.png')
    # Addition of uncracked subimages path lists
    subimglist = subimglist + uncrksubimglist
    # List sorted in ascending order
    sortedlist = sorted(subimglist, key=lambda im: int((im.split('_')[-1]).split('.')[0]))

    # 3.3 Merge sub images (cracked processed and uncracked)
    # # ===============================================================================================================
    final_full_subimage = dict.merge_images(sortedlist, nr, nc)
    # merge sub images with their corresponding label
    final_full_subimage_div = dict.merge_images_with_labels(sortedlist, nr, nc)

    # Save the resulting processed crack and processed crack with subimage divisions and labels
    # # =======================================================================================
    if save_final_full_subimage == True:
        newname = path3.split('\\')[7] + 'MADfullsubimg.png'  # Name for the resulting processed crack
        newname2 = path3.split('\\')[7] + ' divMADfullsubimg.png'  # Name for the resulting processed crack with divisions
        final_full_subimage.save(path3 + '\\' + newname)  # Save image in the corresponding path
        final_full_subimage_div.save(path3 + '\\' + newname2)  # Save image in the corresponding path



end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")