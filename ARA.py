"""Crack width calculation

This code aims to measure the width of the crack obtained in the previous steps. It is based on the the different papers
that perform this measurement generating a skeleton a measuring the length of the perpendicular segment that goes from
edge to edge

Initially based on the GithHub repository: https://github-com.translate.goog/Garamda/Concrete_Crack_Detection_and_Analysis_SW?_x_tr_sl=ko&_x_tr_tl=en&_x_tr_hl=es&_x_tr_pto=wapp
"""

# 4. Preprocess the frame images cropped by crack detection deep learning engine.
#    The preprocess consists of 3 stages.
#   1) Image Binarization : seperate crack section and the noncrack section.
#   2) Skeletonize : extract the central skeleton of the crack.
#   3) Edge detection : extract the edge of the crack.

#   At this stage, Image Binarization will be done.

import matplotlib
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage import data
from skimage.color import rgb2gray
from skimage.data import page
from skimage.filters import (threshold_sauvola)
from PIL import Image
import os
import numpy as np
from scipy import ndimage as ndi
from skimage import feature
from skimage.morphology import skeletonize
from skimage.util import invert
import queue
import math
from skimage import color

def balanced_hist_thresholding(b):
    # Starting point of histogram
    i_s = np.min(np.where(b[0] > 0))
    # End point of histogram
    i_e = np.max(np.where(b[0] > 0))
    # Center of histogram
    i_m = (i_s + i_e) // 2
    # Left side weight
    w_l = np.sum(b[0][0:i_m + 1])
    # Right side weight
    w_r = np.sum(b[0][i_m + 1:i_e + 1])
    # Until starting point not equal to endpoint
    while (i_s != i_e):
        # If right side is heavier
        if (w_r > w_l):
            # Remove the end weight
            w_r -= b[0][i_e]
            i_e -= 1
            # Adjust the center position and recompute the weights
            if ((i_s + i_e) // 2) < i_m:
                w_l -= b[0][i_m]
                w_r += b[0][i_m]
                i_m -= 1
        else:
            # If left side is heavier, remove the starting weight
            w_l -= b[0][i_s]
            i_s += 1
            # Adjust the center position and recompute the weights
            if ((i_s + i_e) // 2) >= i_m:
                w_l += b[0][i_m + 1]
                w_r -= b[0][i_m + 1]
                i_m += 1
    return i_m
# Load an image
path = r'C:\Users\juanc\OneDrive - KTH\Journals\01-Quantification\Image_list'
os.chdir(path)  # Access the path
im = cv2.imread('_DCS6932_195.jpg')#_56_28
img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
window=img[56:56+28,28:56 ]

# Histogram and balanced threshold for window in the initial sub-image
red = np.histogram(window.ravel(), bins=256, range=[0, 256])
trhs = balanced_hist_thresholding(red)
# threshold the image
ret, thresh = cv2.threshold(window, trhs, 255, cv2.THRESH_BINARY)

image=thresh

def CrackWidth(image,pixel_width):
    """
       Calculation of the width of a crack

    Parameters
    ----------
    image : ndarray
        Input data where the crack is located

    Returns
    -------
    list : A list of the different point of the crack with the corresponding width.
    """

    # sauvola_frames_Pw_bw = []
    # sauvola_frames_Pw = []

    cropped_frames = [image]
    widthIM=image.shape[0]
    heightIM=image.shape[1]
    pixel_width = pixel_width

    # Anterior para borrar si esta bien
    # # for i in range(0, len(cropped_frames)):
    # #     img = cropped_frames[i]
    # #     img_gray = rgb2gray(img)
    # #
    # #     # window size and the value of k is the value suggested by the paper 'Concrete Crack Identification Using a UAV Incorporating Hybrid Image Processing'
    # #     # used as is.
    # #
    # #     # window size and k value were used without any changes from the
    # #     # 'Concrete Crack Identification Using a UAV Incorporating Hybrid Image Processing' thesis.
    # #     window_size_Pw = 71
    # #     thresh_sauvola_Pw = threshold_sauvola(img_gray, window_size=window_size_Pw, k=0.42)
    # #
    # #     binary_sauvola_Pw = img_gray > thresh_sauvola_Pw
    # #     binary_sauvola_Pw_bw = img_gray > thresh_sauvola_Pw
    # #
    # #     binary_sauvola_Pw_bw.dtype = 'uint8'
    # #
    # #     binary_sauvola_Pw_bw *= 255
    # #
    # #     # The list which saves the images after image binarization.
    # #
    # #     sauvola_frames_Pw_bw.append(binary_sauvola_Pw_bw)
    # #     sauvola_frames_Pw.append(binary_sauvola_Pw)

    # 5. Extract the skeleton of the crack.

    skeleton_frames_list = []
    skeleton_coord = []

    for i in range(0, len(cropped_frames)):
        # img_Pw = invert(sauvola_frames_Pw[i])
        img_Pw = invert(image)//255
        skeleton_Img = skeletonize(img_Pw)  # generates the skeleton of the image

        skeleton_Img.dtype = 'uint8'  # transform into uint8 data type being numbers from 0 to 255

        skeleton_Img *= 255  # multiply by 255

        skeleton_frames_list.append(skeleton_Img)  # The list which saves the images after the skeleton is obtained.

    # 6. Detect the edges of the crack.

    edges_frames_list = []
    for i in range(0, len(cropped_frames)):
        """ FRom the original code, the edges were found using Canny edge detector but this gave a lot of error in the 
        edges, particularly because in the end it generated vectors with different length than the widths:
        
        edges_Pw = feature.canny(sauvola_frames_Pw[i])  # Detects edges using Canny on the
        edges_Pw.dtype = 'uint8'
        edges_Pw *= 255
        
        """
        # Edges are found using Laplacian. The edges are the negative values and for the sake of the rest of the code
        # they are turned into positive values

        thrlap = cv2.Laplacian(image, cv2.CV_64F)
        for n in range(0, widthIM):
            for m in range(0, heightIM):
                if thrlap[n, m] < 0:
                    thrlap[n, m]=255
                else:
                    thrlap[n, m] = 0

        # The list which saves the images after edge detection.
        edges_Img=thrlap
        edges_frames_list.append(thrlap)

    # 7. Calculate the width of the crack.
    # 1) Find skeleton using BFS
    # 2) Set the direction of the crack by searching skeleton pixels which are 5 pixels away from the skeleton pixel.
    # 3) Draw a perpendicular line of the direction
    # 4) The perpendicular line meets the edge. The distance is calculated by counting pixels on the line.
    # 5) Convert the number of pixels into real mm width, and classify the danger group.

    # dx dir right/left son los pixeles en un circulo alrededor de un pixel central que se estudia. el radio es de 5 pixeles
    # Tiene un problema y es que la grieta se puede pasar justo por las "esquinas" y se puede pasar por alt parte de la infrmacion.
    # Para entender mejr el prblema, ver la imagen del circulo en C:\Users\juanc\OneDrive - KTH\Python\Prueba

    dx_dir_right = [-5, -5, -5, -5, -4, -4, - 3, -3, -2, -1, 0, 1, 2, 3, 3, 4, 4, 5, 5, 5]
    dy_dir_right = [0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 4, 4, 3, 3, 2, 1]

    dx_dir_left = [5, 5, 5, 5, 4, 4, 3, 3, 2, 1, 0, -1, -2, -3, -3, -4, -4, -5, -5, -5]
    dy_dir_left = [0, -1, -2, -3, -3, -4, -4, -5, -5, -5, -5, -5, -5, -5, -4, -4, -3, -3, -2, -1]

    dx_bfs = [-1, -1, 0, 1, 1, 1, 0, -1]
    dy_bfs = [0, 1, 1, 1, 0, -1, -1, -1]

    save_result = []
    save_risk = []

    for k in range(0, len(skeleton_frames_list)):

        # Searching the skeleton through BFS.
        start = [0, 0]  # Start at pixel 0,0
        next = []  # Start an empty vector that will get the list of pixels to check
        q = queue.Queue()  # Defines the queue. Remember, the queue takes elements and once it has studied them it deletes them from the queue
        q.put(start)  # Put the pixel 0,0 at the start of the queue

        len_x = skeleton_frames_list[k].shape[0]  # generates a value equal to the length of the skeleton image
        len_y = skeleton_frames_list[k].shape[1]  # generates a value equal to the height of the skeleton image

        visit = np.zeros((len_x, len_y))  # generates an empty array with the same dimensions that the skeletn img
        crack_width_list = []  # Vector fr the elements of the crack width

        # Find out the direction of the crack from skeleton pixel.
        while (q.empty() == 0):  # Evaluates the next pixel in the queue
            next = q.get()  # next is an element with the info of the next pixel in the queue
            x = next[0]  # vertical position (why the vertical if it is x, I don't know)
            y = next[1]  # horizontal position
            right_x = right_y = left_x = left_y = -1  # initial value for right and left x y

            if (skeleton_frames_list[k][x][y] == 255):  # We check the skeleton in the position x y obtained from the next element in the queue and we see if it is white
                skeleton_coord.append([x,y])  # The coordinates of the skeleton pixel are added to the list
                # Estimating the direction of the crack from skeleton

                # We start checking the pixel in x,y that has a value of 255. We proceed to check the pixels that also have a 255 value in a radius of 5 pixels.
                for i in range(0, len(dx_dir_right)):  # First half of the circle moving from 5pixels up from the x,y pixel and moving right
                    right_x = x + dx_dir_right[i]
                    right_y = y + dy_dir_right[i]
                    if (right_x < 0 or right_y < 0 or right_x >= len_x or right_y >= len_y):  # if we are outside the image's limit, we set righ_x and _y to -1 to move to the next position
                        right_x = right_y = -1
                        continue;
                    if (skeleton_frames_list[k][right_x][right_y] == 255):
                        break;  # Check the place we are, if the skeleton image has something (a value of 1)in the place we are checking or if it is 0. If it has a value of 1, breaks (goes out of the for) and the values for right_x and _y are the distance where something was found
                    if (i == 19):
                        right_x = right_y = -1  # if the count is in the last number, it makes right x and right y equals to -1 to move to the next part of the circle

                if right_x == -1:  # If nothing found, the final value for right x and y is set to the value of x and y
                    right_x = x
                    right_y = y

                for i in range(0, len(dx_dir_left)):  # Second half of the circle moving from 5pixels down from the x,y pixel and moving left
                    left_x = x + dx_dir_left[i]
                    left_y = y + dy_dir_left[i]
                    if (left_x < 0 or left_y < 0 or left_x >= len_x or left_y >= len_y):  # if we are outside the image's limit, we set left_x and _y to -1 to move to the next position
                        left_x = left_y = -1
                        continue;
                    if (skeleton_frames_list[k][left_x][left_y] == 255):
                        break;
                    if (i == 19):
                        left_x = left_y = -1  # if the count is in the last number, it makes left x and left y equals to -1 to move to finish the circle check

                if (left_x == -1):  # final value for left x and y, we set the value to x and y
                    left_x = x
                    left_y = y

                # Set the direction of the crack as angle(theta) by using acos formula
                base = right_y - left_y
                height = right_x - left_x
                hypotenuse = math.sqrt(base * base + height * height)

                if (base == 0 and height != 0):  # If base is 0 BUT height is not 0, it means it is a vertical line thus angle is 90
                    theta = 90.0
                elif (base == 0 and height == 0):  # If base and height are 0, goes back to check the next pixel in the skeleton
                    continue
                else:
                    theta = math.degrees(math.acos((base * base + hypotenuse * hypotenuse - height * height) / (2.0 * base * hypotenuse)))

                theta += 90
                dist = 0

                # Calculate the distance if the perpendicular line meets the edge of the crack.
                for i in range(0, 2):

                    pix_x = x
                    pix_y = y
                    if (theta > 360):
                        theta -= 360
                    elif (theta < 0):
                        theta += 360

                    if (theta == 0.0 or theta == 360.0):  # if the line is horizontal, checks the edge adding 1 to pix_y until it reaches the edge
                        while (1):  # Same as while True
                            pix_y += 1
                            if (pix_y >= len_y):
                                pix_x = x
                                pix_y = y
                                break;
                            if (edges_frames_list[k][pix_x][pix_y] == 255): break;

                    elif (theta == 90.0):  # if the line is vertical, checks the edge subtracting 1 to pix_x until it reaches the edge
                        while (1):
                            pix_x -= 1
                            if (pix_x < 0):
                                pix_x = x
                                pix_y = y
                                break;
                            if (edges_frames_list[k][pix_x][pix_y] == 255): break;

                    elif (theta == 180.0): # if the line is horizontal, checks the edge subtracting 1 to pix_y until it reaches the edge
                        while (1):
                            pix_y -= 1
                            if (pix_y < 0):
                                pix_x = x
                                pix_y = y
                                break;
                            if (edges_frames_list[k][pix_x][pix_y] == 255): break;

                    elif (theta == 270.0):  # if the line is vertical, checks the edge adding 1 to pix_x until it reaches the edge
                        while (1):
                            pix_x += 1
                            if (pix_x >= len_x):
                                pix_x = x
                                pix_y = y
                                break;
                            if (edges_frames_list[k][pix_x][pix_y] == 255): break;
                    else:  # If the angle is other then enters to this section
                        a = 1
                        radian = math.radians(theta)
                        while (1):
                            pix_x = x - round(a * math.sin(radian))
                            pix_y = y + round(a * math.cos(radian))
                            if (pix_x < 0 or pix_y < 0 or pix_x >= len_x or pix_y >= len_y):
                                pix_x = x
                                pix_y = y
                                break;
                            if (edges_frames_list[k][pix_x][pix_y] == 255): break;

                            if (theta > 0 and theta < 90):
                                if (pix_y + 1 < len_y and edges_frames_list[k][pix_x][pix_y + 1] == 255):
                                    pix_y += 1
                                    break;
                                if (pix_x - 1 >= 0 and edges_frames_list[k][pix_x - 1][pix_y] == 255):
                                    pix_x -= 1
                                    break;

                            elif (theta > 90 and theta < 180):
                                if (pix_y - 1 >= 0 and edges_frames_list[k][pix_x][pix_y - 1] == 255):
                                    pix_y -= 1
                                    break;
                                if (pix_x - 1 >= 0 and edges_frames_list[k][pix_x - 1][pix_y] == 255):
                                    pix_x -= 1
                                    break;

                            elif (theta > 180 and theta < 270):
                                if (pix_y - 1 >= 0 and edges_frames_list[k][pix_x][pix_y - 1] == 255):
                                    pix_y -= 1
                                    break;
                                if (pix_x + 1 < len_x and edges_frames_list[k][pix_x + 1][pix_y] == 255):
                                    pix_x += 1
                                    break;

                            elif (theta > 270 and theta < 360):
                                if (pix_y + 1 < len_y and edges_frames_list[k][pix_x][pix_y + 1] == 255):
                                    pix_y += 1
                                    break;
                                if (pix_x + 1 < len_x and edges_frames_list[k][pix_x + 1][pix_y] == 255):
                                    pix_x += 1
                                    break;
                            a += 1  # (creo: If nothing was found with a distance of 1, it increases the distance)

                    dist += math.sqrt((y - pix_y) ** 2 + (x - pix_x) ** 2)
                    theta += 180

                # The list which saves the width of the crack.
                crack_width_list.append(dist)

            for i in range(0, 8):
                next_x = x + dx_bfs[i]
                next_y = y + dy_bfs[i]

                if (next_x < 0 or next_y < 0 or next_x >= len_x or next_y >= len_y): continue;
                if (visit[next_x][next_y] == 0):
                    q.put([next_x, next_y])  # put the evaluated x and y in the queue
                    visit[next_x][next_y] = 1  # NO MUY SEGURO
        """
        # crack_width_list.sort(reverse=True)
        """
        # Convert into real width.
        print(len(crack_width_list))
        if (len(crack_width_list) == 0):
            save_result.append(0)
            real_width = 0
        elif (len(crack_width_list) < 10):
            real_width = round(crack_width_list[len(crack_width_list) - 1] * pixel_width,2)  # the value for the pixel in real life is set, this has to be changed to a variable
            save_result.append(real_width)
        else:
            real_width = round(crack_width_list[9] * pixel_width, 2)
            save_result.append(real_width)

        print('crack width : ', real_width)

        # Classify the danger group.
        if (real_width >= 0.3):
            save_risk.append('high')
            print('risk group : high\n')
        elif (real_width < 0.3 and real_width >= 0.2):
            save_risk.append('medium')
            print('Risk group: medium\n')
        else:
            save_risk.append('low')
            print('Risk group: low\n')


    return crack_width_list,skeleton_coord,skeleton_Img,edges_Img


list,sklcoord,skframes,edgesframes = CrackWidth(image,0.1)
#
#
# widths=np.array(list,sklcoord)
# coordsk=np.array(sklcoord)
# listcomplt=np.column_stack([coordsk,widths])
#
# # # Save those information into text files.
# f1 = open("C:\\Users\\juanc\\OneDrive - KTH\\Python\\Prueba\\listcomplt.txt", 'w')
# # f2 = open("C:\\Users\\juanc\\OneDrive - KTH\\Python\\Prueba\\risk.txt", 'w')
# #
# for z in range(0, len(listcomplt)):
#     f1.write(str(listcomplt[z]) + 'mm' + '\n')
# f1.close()
#
# for z in range(0, len(save_risk)):
#     f2.write(str(save_risk[z]) + '\n')
# f2.close()
