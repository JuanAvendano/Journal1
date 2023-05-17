"""
Created on Fri march 31 2023

@author: jca

Create Histogram  on full img showing the result of using MAD  for a batch.

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


path = r'C:\Users\juanc\OneDrive - KTH\Journals\01-Quantification\Image_list'


pixel_width=0.1
threshold=2.5
full_subimage=False

print_outl = True
save_info = False

for h in range (0,13):
    pathsubfolder = '\Crack ' + str(h + 1)  # Name of the folder where the predicted subimages detected as cracked are located
    path2 = path + pathsubfolder  # Complete the path name with the folder name
    os.chdir(path2)  # Access the path

    #If full subimage seleceted, creates folder for MAD at chosen k and for full subimages
    if full_subimage ==True:
        try:
            os.chdir(path2)  # Access the path
            # Create the full_img folder if it doesn't exist
            os.mkdir(os.path.join(path2, 'MAD k=' + str(threshold)+'full_subimg'))
            print(f'Folder created successfully in {path2}')
        except OSError as e:
            print(f'Error creating folder in {path2}: {str(e)}')


    path3 = path2 + '\MAD k=' + str(threshold) + 'full_subimg'

    img_list = glob.glob(path2 + '\*.png')
    images = [Image.open(path) for path in img_list]

    for i in range(0,len(images)):
        # Convert the image to a numpy array
        images[i] = np.array(images[i])

    cBW= images.copy()
    for i in range(0,len(images)):
        cBW[i]= cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)  #Crack 1 in balck and white



    for i in range(0,len(cBW)):
        # Flattens the image
        WinData=cBW[i].flatten()
        # MAD, clean image, CrackWidth to generate final subimage, GrayScale image with crack highlighted
        resultImg, finalsubimg, low_bound, medlist, Outl=dict.final_subimage(cBW[i],cBW[i],threshold,pixel_width)

        # Save the resulting processed crack
        if save_info == True:
            finalsubimgname =path3.split('\\')[8]+img_list[i].split('\\')[8].split('.')[0]  # Name for the resulting processed crack
            dict.imgSaving(path3, finalsubimgname, finalsubimg)  # Save image in the corresponding path

        if print_outl == True:
            # Generates subimage, histogram, kde, outliers and hist with outliers and median plot
            figure=dict.plot_hist_kde_outl(cBW[i],cBW[i], Outl, medlist)
            figure.show()
            name='histograms'+img_list[i].split('\\')[8].split('.')[0]
            figure.savefig(name+'.png')



end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")