"""
Created on Sat Apr 01 2023
@author: jca

Merge the analyzed subimages (finalsubimg) with uncracked sub images ot obtain the crack stitched together for
visualisation for the different cracks.

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import Dictionary as dict
from skimage.util import invert
from numpy import empty
import time
import glob

# Start time counter
start_time = time.time()
# # ====================================================================================================================
# Inputs
# # ====================================================================================================================
# Path for the different possible cracks
path = r'C:\Users\juanc\OneDrive - KTH\Journals\01-Quantification\Image_list'
# Method used to know where the resulting files need to be saved
method_threshold = 2.5
# If the generated images want to be saved
save_info = True

# region crack info

# Crack dimensions in terms of rows and columns of subimages for each crack (first element is the number of the crack)
crackgeometry = [[1, 1, 7], [2, 5, 2], [3, 2, 8], [4, 9, 3], [5, 5, 3], [6, 5, 3], [7, 4, 6], [8, 5, 2], [9, 5, 4],
                 [10, 6, 3], [11, 6, 8], [12, 6, 2], [13, 6, 2]]

# endregion

# # ====================================================================================================================
# Batch Process
# # ====================================================================================================================
# Number of cracks in the batch to apply the process to
for h in range(0, 13):
    # # ================================================================================================================
    # 1. Paths arrangement
    # # ================================================================================================================
    # Name of the folder where the processed subimages are located
    pathsubfolder = '\Crack ' + str(h + 1)
    # Complete the path name with the folder name
    path2 = path + pathsubfolder
    pathMAD = path2 + '\MAD k=' + str(method_threshold) + 'full_subimg'
    path3 = pathMAD

    # # ================================================================================================================
    # 2. Image with the crack obtained joining the different subimages processed
    # # ================================================================================================================
    # 2.1 Image geometry in terms of subimages
    # # ===============================================================================================================

    # Image with the crack obtained.
    # ===============================================
    # Columns of subimages for the final image
    nc = crackgeometry[h][1]
    # Rows of subimages for the final image
    nr = crackgeometry[h][2]

    # 2.2 List of paths for the cracked processed and uncracked subimages
    # # ===============================================================================================================
    # List of cracked processed subimages (finalsubimg) paths
    subimglist = [os.path.join(path3, f) for f in os.listdir(path3) if "full_subimg" in f]
    # Path where the uncracked sub images are for the current crack
    path4 = path2 + '\\01_Uncracked_subimg\\'
    # List of uncracked subimages paths
    uncrksubimglist = glob.glob(path4 + '*.png')
    # Addition of uncracked subimages paths lists
    subimglist = subimglist + uncrksubimglist
    # List sorted in ascending order
    sortedlist = sorted(subimglist, key=lambda im: int((im.split('_DCS')[-1]).split('.')[0]))

    # 2.3 Merge sub images (cracked processed and uncracked)
    # # ===============================================================================================================
    image = dict.merge_images(sortedlist, nr, nc)
    # Merge sub images with their corresponding label
    image_div = dict.merge_images_with_labels(sortedlist, nr, nc)

    # Save the resulting processed crack and processed crack with subimage divisions and labels
    # =========================================================================================
    if save_info == True:
        newname = path3.split('\\')[7] + 'MAD.png'  # Name for the resulting processed crack
        newname2 = path3.split('\\')[7] + ' divMAD.png'  # Name for the resulting processed crack with divisions
        image.save(path3 + '\\' + newname)  # Save image in the corresponding path
        image_div.save(path3 + '\\' + newname2)  # Save image in the corresponding path

# Finish time counter
end_time = time.time()
# Total time for the code
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
