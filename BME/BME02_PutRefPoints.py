"""
Created on Thr sep 06 2023
@author: jca

Takes the image created in the previous step and adds the ref points

"""



import Dictionary as dict
import os
import matplotlib.pyplot as plt
import cv2
import time


start_time = time.time()

def PutRefPointsBME(n,k,horizontal,saveimg,saveref_points_list):
    # # ================================================================================================================
    # Inputs
    # # ================================================================================================================
    # Crack to check
    n=n
    # Method, used to know where the final subimg is located and where files need to be saved
    method_threshold = k
    # Orientation of the crack
    horizontal=horizontal
    # If the generated final marked image want to be saved
    save_img = saveimg
    # If the generated reference point list want to be saved
    saveref_points_list=saveref_points_list

    # # ===============================================================================================================.
    # 1. Paths arrangement
    # # ===============================================================================================================.
    # path to find the crack img with the reference points is
    path1 = r'C:\Users\jcac\OneDrive - KTH\Journals\01-Quantification\Image_list\Crack ' + str(n)+'\\'
    # Name of the folder where the final img is located
    pathsBME = r'C:\Users\jcac\OneDrive - KTH\Journals\01-Quantification\Image_list\BME\Crack ' + str(n) + '\\k='+ str(method_threshold)+'\\'

    # Access the path
    os.chdir(path1)

    # # ===============================================================================================================.
    # 2. Coordinates
    # # ===============================================================================================================.
    # 2.1 Load and Plot image
    # # ==============================================================================================================..
    # Load the image
    image = cv2.imread('CRACKmarked.png')
    # Transformation from BGR color code to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # 2.2 Mask
    # # ==============================================================================================================..
    # Create a mask for red pixels
    red_mask = cv2.inRange(image, (255, 0, 0), (255, 0, 0))


    # 2.2 Coordinates
    # # ==============================================================================================================..

    # Get the coordinates of red pixels in horizontal order (left to right, top to bottom)
    red_pixels= []
    ref_points_list=[]
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if red_mask[y,x] == 255:
                red_pixels.append((x, y))

    # Columns names
    column_names = ['X coord', 'Y coord'] # X position	Y position
    header = '\t'.join(column_names)
    # Listname
    Listname='ReferencePoints_Crack' + str(n) + '.txt'


    if horizontal==True:
        # Sort the list by the first element (x coordinate)
        red_pixels_horizontal=sorted(red_pixels, key=lambda x: x[0])
        # Display the red pixels
        print('Red pixels in horizontal order:')
        print(red_pixels_horizontal)
        ref_points_list=red_pixels_horizontal

    else:
        # Sort the list by the first element (x coordinate)
        red_pixels_vertical=sorted(red_pixels, key=lambda x: x[1])
        # Display the red pixels
        print('Red pixels in vertical order:')
        print(red_pixels_vertical)
        ref_points_list=red_pixels_vertical

    # 2.3 Save Ref points as a txt
    # # ==============================================================================================================..
    if saveref_points_list == True:
        # Writes header with the names of the columns and fills every row with the values in a certain format
        with open(path1 + '//' + Listname, "w") as output:
            output.write(header + '\n')
            for row in ref_points_list:
                row_str = '\t'.join('{:.2f}'.format(value) for value in row)
                output.write(row_str + '\n')

    # # ===============================================================================================================.
    # 3. Set pixels in final img Crack 1MADfullsubimg
    # # ===============================================================================================================.
    # 3.1 Mark the final image
    # # ==============================================================================================================..
    #Load final subimg
    final_img = cv2.imread(pathsBME+'BME_Crack '+str(n)+'_MAD k='+str(k)+'.png')
    # Transformation from BGR color code to RGB
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

    # Set the pixel from the list red_pixels as green in the final subimage
    for i in range(len(red_pixels)):
        x=red_pixels[i][0]
        y=red_pixels[i][1]
        final_img[y, x] = [0, 255, 0]


    # 3.2 Save the image
    # # ==============================================================================================================..
    if save_img:
        os.chdir(pathsBME)
        # The image is saved in the path
        dict.imgSaving(pathsBME, 'finalimg_marked_BME', final_img)

#
# # # ====================================================================================================================
# # Inputs
# # # ====================================================================================================================
# # Method, used to know where the final subimg is located and where files need to be saved
listk=[2]
for x in listk:
    k = x
    # Cracks to be studied
    start=1
    end=29


    # If the generated final marked image want to be saved
    save_img = False
    # If the generated reference point list want to be saved
    saveref_points_list = False

    # Batch process
    for i in range(start, end+1):
        horizontal = True
        if i in [1, 3, 7, 11,14,15,16,17,18,19,20,21,22,23,24,25,26]:
            horizontal = False
        PutRefPointsBME(i, k, horizontal, save_img, saveref_points_list)


# Finish time counter
end_time = time.time()
# Total time for the code
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")