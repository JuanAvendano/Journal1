"""
Created on Wed aug 23 2023

@author: jca


Process on a single crack to obtain final subimages (with crack, edges and skeleton pixels), and final images
reconstructed with and without divisions using Bilateral filtering-MAD thresholding-Laplacian edge detection- Dilation


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

start_time = time.time()

def unit_result_bilateral_MAD(n, k, pixelwidth, save_subimg,save_info, saveimgparts, save_preliminaryimg, saveimg):
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
    # If skeletons, edges want to be saved as png
    save_img_parts = saveimgparts
    # If the generated complete crack pixels and crack pixels with division want to be saved
    save_preliminaryimg = save_preliminaryimg
    # If the generated final images want to be saved
    save_img = saveimg

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
    pathBML = path + '\BML'

    try:
        os.chdir(pathBML)  # Access the path
        # Create the Crack folder if it doesn't exist
        os.mkdir(pathBML+'\Crack ' + str(n))
        print(f'Folder created successfully in {pathBML}')
    except OSError as e:
        print(f'Error creating folder in {pathBML}: {str(e)}')
    try:
        os.mkdir(pathBML+'\Crack ' + str(n)+'\\k='+ str(method_threshold))
        print(f'Folder created successfully in {pathBML}')
    except OSError as e:
        print(f'Error creating folder in {pathBML}: {str(e)}')


    pathResult = pathBML+'\Crack ' + str(n)+ '\\k='+str(method_threshold)


    # 2. Cracked and uncracked sub-images lists
    # # ============================================================================================================
    # Access the path for Cracked sub-images
    os.chdir(path)

    # List of cracked  subimages paths
    crackedimglist0 = os.listdir(pathCracked)
    crackedimglist = [element.split('.')[0] for element in crackedimglist0 if element.endswith('.png')]

    # List of cracked  subimages paths
    uncrackedimglist = os.listdir(pathunCracked)


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

        # 3. Bilateral filtering
        # # ============================================================================================================

        BilateralImg,Databilateral=dicc.bilateral_filter(imageBW,method_threshold)

        # 4. MAD and threshold
        # # ============================================================================================================
        Outl, med, medlist, inliers, trhs = dicc.detect_outliers_mad(Databilateral, method_threshold)
        ret, MADImg = cv2.threshold(BilateralImg, trhs, 255, cv2.THRESH_TRUNC)

        # 5. Erosion
        # # ============================================================================================================
        ErodedImg= dicc.erosion(MADImg)

        # 6. Edge detection with Laplacian
        # # ============================================================================================================
        LaplacianImg= cv2.Laplacian(ErodedImg, cv2.CV_64F)


        neg_mask_LoGbilateral = LaplacianImg < 0
        pos_mask_filled = ErodedImg <np.floor(trhs)


        width = LaplacianImg.shape[0]
        height = LaplacianImg.shape[1]
        finalsubimg = empty([width, height, 3], dtype=np.uint8)
        for x in range(0, width):
            for y in range(0, height):
                if pos_mask_filled[x, y] and LaplacianImg[x, y] < 0:
                    finalsubimg[x, y] = [255, 0, 0]  # If pixel is part of skeleton paint red
                elif pos_mask_filled[x, y]:
                    finalsubimg[x, y] = [255, 255, 0]  # If pixel is part of crack paint yellow
                elif LaplacianImg[x, y] < 0:
                    finalsubimg[x, y] = [0, 0, 255]  # If pixel is part of edge, paint blue
                else:
                    finalsubimg[x, y] = 0

        CrackPixels = cv2.inRange(finalsubimg, (255, 255, 0), (255, 255, 0))

        #
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(EdgePixels, cmap='gray')
        # plt.subplot(122)
        # plt.imshow(CrackPixels, cmap='gray')
        #
        #
        # plt.figure('Masks')
        # plt.subplot(231)
        # plt.title('bilateral')
        # plt.imshow(BilateralImg, cmap='gray')
        # plt.subplot(232)
        # plt.title('bilateral_truncated')
        # plt.imshow(MADImg, cmap='gray')
        # plt.subplot(233)
        # plt.title('bilateral_truncated_filled ')
        # plt.imshow(ErodedImg, cmap='gray')
        # plt.subplot(234)
        # plt.title('LoGbilateral')
        # plt.imshow(LaplacianImg, cmap='gray')
        # plt.subplot(235)
        # plt.title('neg_mask_LoGbilateral ')
        # plt.imshow(neg_mask_LoGbilateral, cmap='gray')
        # plt.subplot(236)
        # plt.title('pos_mask_filled')
        # plt.imshow(pos_mask_filled, cmap='gray')
        #
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(finalsubimg)
        # plt.subplot(122)
        # plt.imshow(prueba)
        # plt.show(block=False)
        #
        # print('dg')

        resultList.append((img_to_select,CrackPixels))

        CrackPixelsname= 'CrackPixels' + img_to_select
        if save_subimg:
            os.chdir(pathResult)
            # The image is saved in the path
            dicc.imgSaving(pathResult, CrackPixelsname, CrackPixels)



    black_subImg = empty([width, height])
    for j in range(0, len(uncrackedimglist)):

        resultList.append((uncrackedimglist[j],black_subImg))

    # 7. Create the complete image
    # # ============================================================================================================
    sortedlist = sorted(resultList, key=lambda element: int((element[0].split('_')[-1]).split('.')[0]))
    # Columns of subimages for the final image
    nc = crackgeometry[n-1][1]
    # Rows of subimages for the final image
    nr = crackgeometry[n-1][2]

    just_images_sorted_list=[img[1] for img in sortedlist]
    image = dicc.merge_images(just_images_sorted_list, nr, nc)
    # merge sub images with their corresponding label
    image_div = dicc.merge_images_with_labels(sortedlist, nr, nc)
    if save_preliminaryimg == True:
        newname = pathResult.split('\\')[8] + 'BML.png'  # Name for the resulting processed crack
        newname2 = pathResult.split('\\')[8] + ' divBML.png'  # Name for the resulting processed crack with divisions
        dicc.imgSaving(pathResult, newname, image)
        dicc.imgSaving(pathResult, newname2, image_div)

    # 8. Erosion
    # # ============================================================================================================
    image= dicc.erosion(image)

    # 9. Obtain final edges and skeletons
    # # ============================================================================================================
    # Access the path
    # os.chdir(pathResult)

    widths, coordsk, skframes, edgesframes, completeList = dicc.CrackWidth(image // 255, pixel_width)

    # Names for the results images

    skframesname = 'skframes_' + 'Crack ' + str(n)  # name of the skeleton image that will be saved
    edgesframesname = 'edgesframes_' + 'Crack ' + str(n)  # name of the edges image that will be saved
    completeListname = 'completeList_' + 'Crack ' + str(n) + '.txt'  # name of the image that will be saved

    # Saves the image without small object, skeletons, edges
    # ===============================================
    if save_img_parts == True:
        os.chdir(pathResult)
        # the image where skeleton is saved in the path
        dicc.BinarySaving(pathResult, skframesname, skframes)
        # the image where edges of the crack is saved in the path
        dicc.BinarySaving(pathResult, edgesframesname, edgesframes)

    # saves the list as a txt file
    # ===============================================
    if save_info == True:
        # Columns names
        column_names = ['Y coord', 'X coord', 'Width (pxl)', 'Width (mm)', 'Danger group']
        header = '\t'.join(column_names)
        # Writes header with the names of the columns and fills every row with the values in a certain format
        with open(pathResult  +'\\'+ completeListname, "w") as output:
            output.write(header + '\n')
            for row in completeList:
                row_str = '\t'.join('{:.2f}'.format(value) for value in row)
                output.write(row_str + '\n')

    #
    # Colors over the subimage, the obtained skeleton, crack pixels and edges.
    # ========================================================================================================
    width = image.shape[0]
    height = image.shape[1]
    finalImg = empty([width, height, 3], dtype=np.uint8)  # creates the image with the obtained crack .
    for x in range(0, width):
        for y in range(0, height):
            if skframes[x, y] > 0:
                finalImg[x, y] = [255, 0, 0]  # If pixel is part of skeleton paint red
            elif image[x, y] > 0:
                finalImg[x, y] = [255, 255, 0]  # If pixel is part of crack paint yellow
            elif edgesframes[x, y] > 0:
                finalImg[x, y] = [0, 0, 255]  # If pixel is part of edge, paint blue

            else:
                finalImg[x, y] = InitialCrackImage[x, y]

    # Saves the final subimage as a png if save_img selected
    # ======================================================
    # Name of the image that will be saved
    finalImgName = 'finalImg' + 'Crack ' + str(n)
    if save_img:
        os.chdir(path)
        # The image is saved in the path
        dicc.imgSaving(pathResult, finalImgName, finalImg)

unit_result_bilateral_MAD(22, 3.5, 0.01,save_subimg=False, save_info=True, saveimgparts=False, save_preliminaryimg=True, saveimg=True)

# Finish time counter
end_time = time.time()
# Total time for the code
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")