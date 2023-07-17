"""
Created on Mon Jul 10 2023
@author: jca

"""



import Dictionary as dict
import os
import matplotlib.pyplot as plt
from numpy import empty
import time
import cv2
import numpy as np

start_time = time.time()

def skeleton_genral(n, k, pixelwidth,  saveimg,saveinfo):
    # # ================================================================================================================
    # Inputs
    # # ================================================================================================================
    # Path for the different possible cracks
    path = r'C:\Users\juanc\OneDrive - KTH\Journals\01-Quantification\Image_list'
    # Number of the cracked that is going to be processed
    n = n
    # Must be 0 if method is Balanced histogram. If it is MAD the value is the threshold value
    method_threshold = k
    # If the info related to x,y coordinates and widths want to be saved as text file
    save_img = saveimg

    save_info=saveinfo
    # pixel_width in mm
    pixel_width = pixelwidth

    # # ============================================================================================================
    # 1. Paths arrangement
    # # ============================================================================================================
    # Name of the folder where the information of the desired crack is located
    pathsubfolder = '\Crack ' + str(n)
    path2 = path + pathsubfolder  # Complete the path name with the folder name
    # Path where results will be saved if using MAD
    pathMAD = path2 + '\MAD k=' + str(method_threshold)+' full_subimg'
    os.chdir(pathMAD)
    # Get the subimage (selected image) and turns it into greyscale (imageBW)
    # ========================================================================================================
    selectedimage, imageBW = dict.selectimg('Crack '+str(n)+'MADfullsubimg')

    # Create a mask for red pixels
    yellow_mask = cv2.inRange(selectedimage, (0, 255, 255), (0, 255, 255))
    blue_mask = cv2.inRange(selectedimage, (0, 0, 255), (0, 0, 255))
    crack_mask = yellow_mask+blue_mask

    # Get the widths, coordinate of the skeleton, img with skeleton,img with edges, info list
    widths, coordsk, skframes, edgesframes, completeList = dict.CrackWidth(crack_mask // 255, pixel_width)

    # Sets the image width and heigh
    width = selectedimage.shape[0]
    height = selectedimage.shape[1]

    # Creates the result image
    resultImg = selectedimage.copy() * 0



    # Creates the finalsubimage element that will have the crack highlighted and the rest of the pixel in grayscale
    finalsubimg = empty([width, height, 3], dtype=np.uint8)

    for x in range(0, width):
        for y in range(0, height):
            if skframes[x, y] > 0:
                finalsubimg[x, y] = [255, 0, 0]  # If pixel is part of skeleton paint red

            elif crack_mask[x, y] > 0:
                finalsubimg[x, y] = [255, 255, 0]  # If pixel is part of crack paint yellow

            elif edgesframes[x, y] > 0:
                finalsubimg[x, y] = [0, 0, 255]  # If pixel is part of edge, paint blue
            else:
                finalsubimg[x, y] = selectedimage[x, y]

    # Saves the final subimage as a png if save_img selected
    # ======================================================
    # Name of the image that will be saved
    finalsubimgname = '000_Crack '+str(n)+'_MAD k=' + str(method_threshold)+' full_subimg'
    if save_img:
        os.chdir(pathMAD)
        # The image is saved in the path
        dict.imgSaving(pathMAD, finalsubimgname, finalsubimg)
    # saves the list as a txt file
    # ===============================================
    completeListname = '000_completeList_Crack' + str(n)+' full_subimg' + '.txt'  # name of the image that will be saved
    if save_info == True:
        # Columns names
        column_names = ['Y coord', 'X coord', 'Width (pxl)', 'Width (mm)', 'Danger group']
        header = '\t'.join(column_names)
        # Writes header with the names of the columns and fills every row with the values in a certain format
        with open(pathMAD + '//' + completeListname, "w") as output:
            output.write(header + '\n')
            for row in completeList:
                # Swap the values of the second and first columns
                row[0], row[1] = row[1], row[0]
                row_str = '\t'.join('{:.2f}'.format(value) for value in row)
                output.write(row_str + '\n')


# # ====================================================================================================================
# Inputs
# # ====================================================================================================================
listk=[2.2,3,3.5]
for x in listk:
    # Must be 0 if method is Balanced histogram. If it is MAD the value is the threshold value
    k = x
    # Cracks to be studied
    start = 1
    end = 29
    save_img = True
    # If the merged list of x,y coordinates and widths want to be saved as text file
    save_info=True
    # Batch process
    for i in range(start, end + 1):
        skeleton_genral(i, k, 0.08,save_img,save_info)

# Finish time counter
end_time = time.time()
# Total time for the code
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")