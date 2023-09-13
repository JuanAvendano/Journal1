"""
Created on Wed sep 09 2023

@author: jca


Process on a single crack to obtain final subimages (with crack, edges and skeleton pixels), and final images
reconstructed with and without divisions using Bilateral filtering-MAD thresholding-Erosion-Laplace


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
import Dictionary as dicc
from scipy.stats import norm
import glob


start_time = time.time()

def unit_result_BME(n, k, pixelwidth, save_subimg,save_info, save_FINAL_img):
    # # ================================================================================================================
    # Inputs
    # # ================================================================================================================

    # Number of the cracked that is going to be processed
    n = n

    # Must be 0 if method is Balanced histogram. If it is MAD the value is the threshold value
    method_threshold = k
    # pixel_width in mm
    pixel_width = pixelwidth
    # If the generated subimages want to be saved
    save_subimg=save_subimg
    # If the info related to x,y coordinates and widths want to be saved as text file
    save_info = save_info

    # If the generated final images want to be saved
    save_FINAL_img = save_FINAL_img

    # # ================================================================================================================
    # Crack info
    # # ================================================================================================================

    # Crack dimensions in terms of rows and columns of subimages for each crack (first element is the number of the
    # crack, second number is the number of columns and third is rows)

    crackgeometry = [[1, 1, 7], [2, 5, 2], [3, 2, 8], [4, 9, 3], [5, 5, 3], [6, 5, 3], [7, 4, 6], [8, 5, 2], [9, 5, 4],
                     [10, 6, 3], [11, 6, 8], [12, 6, 2], [13, 6, 2], [14,8,10],[15,7,8],[16,4,18],[17,3,9],[18,4,13],
                     [19,3,10],[20,3,7],[21,5,11],[22,6,9],[23,4,13],[24,5,17],[25,6,16],[26,4,10],[27,6,4],[28,17,3],
                     [29,15,4]]


    # # ================================================================================================================
    # Process
    # # ================================================================================================================
    # 1. Paths arrangement
    # # ============================================================================================================


    # Path for the different possible cracks
    path = r'C:\Users\jcac\OneDrive - KTH\Journals\01-Quantification\Image_list'
    pathCrak = path +'\Crack ' + str(n) + '\\'
    pathCracked = pathCrak + '00_Cracked_subimg\\'
    pathunCracked = pathCrak + '01_Uncracked_subimg\\'
    pathBME = path + '\BME'

    try:
        os.chdir(pathBME)  # Access the path
        # Create the Crack folder if it doesn't exist
        os.mkdir(pathBME+'\Crack ' + str(n))
        print(f'Folder created successfully in {pathBME}')
    except OSError as e:
        print(f'Error creating folder in {pathBME}: {str(e)}')
    try:
        os.mkdir(pathBME+'\Crack ' + str(n)+'\\k='+ str(method_threshold))
        print(f'Folder created successfully in {pathBME}')
    except OSError as e:
        print(f'Error creating folder in {pathBME}: {str(e)}')


    pathResult = pathBME+'\Crack ' + str(n)+ '\\k='+str(method_threshold)


    # 2. Cracked and uncracked sub-images lists
    # # ============================================================================================================
    # Access the path for Cracked sub-images
    os.chdir(path)

    # List of cracked  subimages paths
    crackedimglist0 = os.listdir(pathCracked)
    crackedimglist = [element.split('.')[0] for element in crackedimglist0 if element.endswith('.png')]

    # List of cracked  subimages paths
    uncrackedimglist =[os.path.join(pathunCracked, f) for f in os.listdir(pathunCracked) if "DCS" in f and ".png" in f]



    resultList=[]
    InitialCrackImage,InitialCrackImageBW=dicc.selectimg(os.path.join(pathCrak,'CRACK'))

    # process over the different sub-images
    # # ============================================================================================================
    for i in range(0, len(crackedimglist)):

        # Access the path for Cracked sub-images
        os.chdir(pathCracked)
        img_to_select=crackedimglist[i]
        # Get the subimage (selected image) and turns it into greyscale (imageBW)
        selectedimage, imageBW = dicc.selectimg(img_to_select)


        resultImg, finalsubImg, low_bound, medlist, Outl = dicc.final_subimage_dilation(imageBW, selectedimage, method_threshold,
                                                                               pixel_width)

        # Save the resulting processed crack
        if save_subimg == True:
            finalFull_imgname ='BME_'+pathResult.split('\\')[9]+crackedimglist[i]  # Name for the resulting processed crack
            dicc.imgSaving(pathResult, finalFull_imgname, finalsubImg)  # Save image in the corresponding path

    # # ================================================================================================================
    # 3. Image with the crack obtained joining the different subimages processed
    # # ================================================================================================================
    # 3.1 Image geometry in terms of subimages
    # # ===============================================================================================================

    # Columns of subimages for the final image
    nc = crackgeometry[n - 1][1]
    # Rows of subimages for the final image
    nr = crackgeometry[n - 1][2]

    # 3.2 List of paths for the cracked processed and uncracked subimages
    # # ===============================================================================================================
    # List of cracked processed subimages (finalsubimg) paths
    subimglist = [os.path.join(pathResult, f) for f in os.listdir(pathResult) if "DCS" in f and ".png" in f]
    # List of uncracked subimages paths
    uncrksubimglist = uncrackedimglist
    # Addition of uncracked subimages path lists
    subimglist = subimglist + uncrksubimglist
    # List sorted in ascending order
    sortedlist = sorted(subimglist, key=lambda im: int((im.split('_')[-1]).split('.')[0]))

    # 3.3 Merge sub images (cracked processed and uncracked)
    # # ===============================================================================================================
    full_image = dicc.merge_images(sortedlist, nr, nc)
    full_image=np.array(full_image)
    # merge sub images with their corresponding label
    full_image_div = dicc.merge_images_with_labels(sortedlist, nr, nc)
    full_image_div=np.array(full_image_div)

    # Create a mask for red pixels
    yellow_mask = cv2.inRange(full_image, (255, 255, 0), (255, 255, 0))
    red_mask = cv2.inRange(full_image, (255, 0, 0), (255, 0, 0))
    crack_mask = yellow_mask + red_mask

    # Get the widths, coordinate of the skeleton, img with skeleton,img with edges, info list
    widths, coordsk, skframes, edgesframes, completeList = dicc.CrackWidth(crack_mask // 255, pixel_width)

    # Sets the image width and heigh
    width = full_image.shape[0]
    height = full_image.shape[1]


    # Creates the finalsubimage element that will have the crack highlighted and the rest of the pixel in grayscale
    finalImg = empty([width, height, 3], dtype=np.uint8)

    red_pixels = skframes > 0
    yellow_pixels = crack_mask > 0
    blue_pixels = edgesframes > 0


    finalImg[yellow_pixels] = [255, 255, 0]
    finalImg[red_pixels] = [255, 0, 0]
    finalImg[blue_pixels] = [0, 0, 255]
    finalImg[~(red_pixels | yellow_pixels | blue_pixels)] = InitialCrackImage[~(red_pixels | yellow_pixels | blue_pixels)]


    # Saves the final image as a png if save_FINAL_img selected
    # ======================================================
    # Name of the image that will be saved
    finalFull_imgname = 'BME_Crack ' + str(n) + '_MAD k=' + str(method_threshold)
    if save_FINAL_img:
        os.chdir(pathResult)
        # The image is saved in the path
        dicc.imgSaving(pathResult, finalFull_imgname, finalImg)
    # saves the list as a txt file
    # ===============================================
    completeListname = 'BME_completeList_Crack' + str(
        n) + ' full_subimg' + '.txt'  # name of the image that will be saved
    if save_info == True:
        # Columns names
        column_names = ['Y coord', 'X coord', 'Width (pxl)', 'Width (mm)', 'Danger group']
        header = '\t'.join(column_names)
        # Writes header with the names of the columns and fills every row with the values in a certain format
        with open(pathResult + '//' + completeListname, "w") as output:
            output.write(header + '\n')
            for row in completeList:
                # Swap the values of the second and first columns
                row[0], row[1] = row[1], row[0]
                row_str = '\t'.join('{:.2f}'.format(value) for value in row)
                output.write(row_str + '\n')


listk=[2]
for x in listk:
    k = x
# # Cracks to be studied
    start = 1
    end = 29
    for i in range(start, end + 1):
        unit_result_BME(i, k, 0.1,save_subimg=False, save_info=True,   save_FINAL_img=False)

# Finish time counter
end_time = time.time()
# Total time for the code
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")