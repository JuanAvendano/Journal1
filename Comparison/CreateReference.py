"""
Created on Wed sep 11 2023

@author: jca


Creates the reference images that are manual pixel based clasification, it takes as base the results from the sauvola
process. This is process 1 (the folders have been change and have to be updated)
Process 2: takes the clean images, the base images and overlaps them to create an image where the crack pixel are green
and it is saved. the image can then be opened in paint to complete the green regions as needed.
Process 3: after the images are completed and gaps are filled as green pixels, we upload them to create a mask and
generate only images in black and white to which we can compare the results of the methods.
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

process1=False
process2= False
process3= True



# If the generated final images want to be saved
save_FINAL_img = True
# Process 1: takes the initial results for a method (sauvola for example) and turns all the yellow and red pixels into
# black and background as white and saves it for later cleaning (that part is done manually)
if process1==True:
    # # ================================================================================================================
    # Process
    # # ================================================================================================================
    # 1. Paths arrangement
    # # ============================================================================================================


    # Path for the different possible cracks
    path = r'C:\Users\jcac\OneDrive - KTH\Journals\01-Quantification\Image_list\Sauvola\Sauvola results'
    pathResult1 =  r'C:\Users\jcac\OneDrive - KTH\Journals\01-Quantification\Image_list\Sauvola\Sauvola results'


    # Access the path
    os.chdir(path)

    # List of images paths
    crackedimglist0 = os.listdir(path)
    crackedimglist = [element.split('.')[0] for element in crackedimglist0 if element.endswith('.png')]





    # process over the different sub-images
    # # ============================================================================================================
    for i in range(0, len(crackedimglist)):

        os.chdir(path)
        img_to_select=crackedimglist[i]
        # Get the subimage (selected image) and turns it into greyscale (imageBW)
        selectedimage, imageBW = dicc.selectimg(img_to_select)
        selectedimage=cv2.cvtColor(selectedimage,cv2.COLOR_BGR2RGB)
        # Create a mask for red pixels
        yellow_mask = cv2.inRange(selectedimage, (255, 255, 0), (255, 255, 0))
        red_mask = cv2.inRange(selectedimage, (255, 0, 0), (255, 0, 0))
        crack_mask = yellow_mask + red_mask



        # Sets the image width and heigh
        width = selectedimage.shape[0]
        height = selectedimage.shape[1]


        # Creates the finalsubimage element that will have the crack as white and the rest of the pixels in black
        finalImg = empty([width, height, 3], dtype=np.uint8)


        crack_pixels = crack_mask > 0
        finalImg[crack_pixels] = [0,0,0]
        finalImg[~( crack_pixels)] = [255, 255, 255]




        # Saves the final image as a png if save_FINAL_img selected
        # ======================================================
        # Name of the image that will be saved
        cracknumb=img_to_select.split('CRACK')[-1]
        finalFull_imgname = 'REF_Crack ' + cracknumb
        if save_FINAL_img:
            os.chdir(pathResult1)
            # The image is saved in the path
            dicc.imgSaving(pathResult1, finalFull_imgname, finalImg)

# Process 2: after the cleaning of blobs, it takes the clean result and puts it as green pixels on top of the base
# images that have no marking so one can check if there are gaps or regions missing to paint as green (that part is done
# manually)
if process2==True:
    path = r'C:\Users\jcac\OneDrive - KTH\Journals\01-Quantification\Image_list'

    pathREF2=r'C:\Users\jcac\OneDrive - KTH\Journals\01-Quantification\Image_list\Reference images'
    pathResults2=r'C:\Users\jcac\OneDrive - KTH\Journals\01-Quantification\Image_list\Reference areas\\'


    start=1
    end=29
    for n in range(start,end+1):
        # List of images paths
        pathImages = r'C:\Users\jcac\OneDrive - KTH\Journals\01-Quantification\Image_list\Reference areas' + '\\'
        os.chdir(pathImages)
        img_to_select = 'CRACK'+str(n)
        selectedimage, imageBW = dicc.selectimg(img_to_select)
        selectedimage=cv2.cvtColor(selectedimage,cv2.COLOR_BGR2RGB)

        os.chdir(pathREF2)
        img_to_select = 'REF_Crack '+str(n)
        REFimage, REFimageBW = dicc.selectimg(img_to_select)

        black_mask = invert(REFimageBW) >0
        final_Ref_img=np.copy(selectedimage)
        final_Ref_img[black_mask]= [0,255,0]

        # Saves the final image as a png if save_FINAL_img selected
        # ======================================================
        # Name of the image that will be saved
        finalFull_imgname = 'final_'+img_to_select
        if save_FINAL_img:
            os.chdir(pathResults2)
            # The image is saved in the path
            dicc.imgSaving(pathResults2, finalFull_imgname, final_Ref_img)


# Process 3: after the images are completed and gaps are filled as green pixels, we upload them to create a mask and
# generate only images in black and white to which we can compare the results of the methods.
if process3==True:
    path = r'C:\Users\jcac\OneDrive - KTH\Journals\01-Quantification\Image_list'

    pathREF3=r'C:\Users\jcac\OneDrive - KTH\Journals\01-Quantification\Image_list\Reference\Reference areas'
    pathResults3=r'C:\Users\jcac\OneDrive - KTH\Journals\01-Quantification\Image_list\Reference\Final references masks\\'


    start=1
    end=29
    for n in range(start,end+1):
        # List of images paths
        pathImages = pathREF3
        os.chdir(pathImages)
        img_to_select = 'final_REF_Crack '+str(n)
        selectedimage, imageBW = dicc.selectimg(img_to_select)
        selectedimage=cv2.cvtColor(selectedimage,cv2.COLOR_BGR2RGB)

        # Create a mask for red pixels
        green_mask = cv2.inRange(selectedimage, (0, 255, 0), (0, 255, 0))
        crack_mask = green_mask

        # Sets the image width and heigh
        width = selectedimage.shape[0]
        height = selectedimage.shape[1]

        # Creates the finalsubimage element that will have the crack as white and the rest of the pixels in black
        final_REF_mask = empty([width, height, 3], dtype=np.uint8)

        crack_pixels = crack_mask > 0
        final_REF_mask[crack_pixels] = [0, 0, 0]
        final_REF_mask[~(crack_pixels)] = [255, 255, 255]


        # Saves the final image as a png if save_FINAL_img selected
        # ======================================================
        # Name of the image that will be saved
        cracknumb = img_to_select.split('Crack')[-1]
        finalFull_imgname = 'Final_REF_Mask ' + cracknumb
        if save_FINAL_img:
            os.chdir(pathResults3)
            # The image is saved in the path
            dicc.imgSaving(pathResults3, finalFull_imgname, final_REF_mask)

# Finish time counter
end_time = time.time()
# Total time for the code
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")