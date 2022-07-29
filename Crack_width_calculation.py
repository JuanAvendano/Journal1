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

sauvola_frames_Pw_bw = []
sauvola_frames_Pw = []

# Load an image
path = r'C:\Users\juanc\OneDrive - KTH\Python\Prueba\02.1-Cracked_predicted'
os.chdir(path)  # Access the path
image = cv2.imread('_DCS6695_412 - Copy.jpg')
cropped_frames = [image]

pixel_width=0.1

for i in range(0, len(cropped_frames)):
    img = cropped_frames[i]
    img_gray = rgb2gray(img)

    # window size and the value of k is the value suggested by the paper 'Concrete Crack Identification Using a UAV Incorporating Hybrid Image Processing'
    # used as is.

    # window size and k value were used without any changes from the
    # 'Concrete Crack Identification Using a UAV Incorporating Hybrid Image Processing' thesis.
    window_size_Pw = 71
    thresh_sauvola_Pw = threshold_sauvola(img_gray, window_size=window_size_Pw, k=0.42)

    binary_sauvola_Pw = img_gray > thresh_sauvola_Pw
    binary_sauvola_Pw_bw = img_gray > thresh_sauvola_Pw

    binary_sauvola_Pw_bw.dtype = 'uint8'

    binary_sauvola_Pw_bw *= 255

    # The list which saves the images after image binarization.

    sauvola_frames_Pw_bw.append(binary_sauvola_Pw_bw)
    sauvola_frames_Pw.append(binary_sauvola_Pw)

# 5. Extract the skeleton of the crack.



skeleton_frames_Pw = []

for i in range(0, len(cropped_frames)):
    img_Pw = invert(sauvola_frames_Pw[i])

    skeleton_Pw = skeletonize(img_Pw)

    skeleton_Pw.dtype = 'uint8'

    skeleton_Pw *= 255

    # The list which saves the images after the skeletonization.
    skeleton_frames_Pw.append(skeleton_Pw)

# 6. Detect the edges of the crack.



edges_frames_Pw = []
edges_frames_Pl = []

for i in range(0, len(cropped_frames)):
    edges_Pw = feature.canny(sauvola_frames_Pw[i], 0.09)

    edges_Pw.dtype = 'uint8'

    edges_Pw *= 255

    # The list which saves the images after edge detection.
    edges_frames_Pw.append(edges_Pw)

# 7. Calculate the width of the crack.
# 1) Find skeleton using BFS
# 2) Set the direction of the crack by searching skeleton pixels which are 5 pixels away from the skeleton pixel.
# 3) Draw a perpendicular line of the direction
# 4) The perpendicular line meets the edge. The distance is calculated by counting pixels on the line.
# 5) Convert the number of pixels into real mm width, and classify the danger group.


# dx dir right/left son los pixeles en un circulo alrededor de un pixel central que se estudia. el radio es de 5 pixeles
# Tiene un problema y es que la grieta se puede pasar justo por las "esquinas" y se puede pasar por alt parte de la infrmacion.
# Para entender mejr el prblema, ver la imagen del circulo en C:\Users\juanc\OneDrive - KTH\Python\Prueba

dx_dir_right = [-5, -5, -5, -5, -4, -4 - 3, -3, -2, -1, 0, 1, 2, 3, 3, 4, 4, 5, 5, 5]
dy_dir_right = [0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 4, 4, 3, 3, 2, 1]

dx_dir_left = [5, 5, 5, 5, 4, 4, 3, 3, 2, 1, 0, -1, -2, -3, -3, -4, -4, -5, -5, -5]
dy_dir_left = [0, -1, -2, -3, -3, -4, -4, -5, -5, -5, -5, -5, -5, -5, -4, -4, -3, -3, -2, -1]

dx_bfs = [-1, -1, 0, 1, 1, 1, 0, -1]
dy_bfs = [0, 1, 1, 1, 0, -1, -1, -1]

save_result = []
save_risk = []

for k in range(0, len(skeleton_frames_Pw)):

    # Searching the skeleton through BFS.
    start = [0, 0]  #Start at pixel 0,0
    next = []       #Start an empty vector that will get the list of pixels to check
    q = queue.Queue()   #Defines the queue. Remember, the queue takes elements and once it has studied them it deletes them frm the queue
    q.put(start)    # Put the pixel 0,0 at the start of the queue

    len_x = skeleton_frames_Pw[k].shape[0]  # generates a value equal to the length of the skeleton image
    len_y = skeleton_frames_Pw[k].shape[1]  # generates a value equal to the height of the skeleton image

    visit = np.zeros((len_x, len_y))        # generates an empty array with the same dimensions that the skeletn img
    crack_width_list = []   #Vectr fr the elements of the crack width

    # Find out the direction of the crack from skeleton pixel.
    while (q.empty() == 0): # Evaluates the next pixel in the queue
        next = q.get()  # next is an element with the info of the next pixel in the queue
        x = next[0]     # vertical position (why the vertical if it is x, I don't know)
        y = next[1]     # vertical position
        right_x = right_y = left_x = left_y = -1    #initial value for right and left x y

        if (skeleton_frames_Pw[k][x][y] == 255): # We check the skeleton in the position x y obtained from the next element in the queue and we see if it is white

            # Estimating the direction of the crack from skeleton

            # We start checking the pixel in x,y that has a value of 255. We proceed to check the pixels that also have a 255 value in a radius of 5 pixels.
            for i in range(0, len(
                    dx_dir_right)):  # First half of the circle moving from 5pixels up from the x,y pixel and moving right
                right_x = x + dx_dir_right[i]
                right_y = y + dy_dir_right[i]
                if (
                        right_x < 0 or right_y < 0 or right_x >= len_x or right_y >= len_y):  # if we are outside the image's limit, we set righ_x and _y to -1 to move to the next position
                    right_x = right_y = -1
                    continue;
                if (skeleton_frames_Pw[k][right_x][
                    right_y] == 255): break;  # Check the place we are if the skeleton image has something (a value of 1)in the place we are checking or if it is 0. If it has a value of 1, breaks (goes out of the for) and the values for right_x and _y are the distance where something was found
                if (
                        i == 13): right_x = right_y = -1  # if the count is in the last number, it makes right x and right y equals to -1 to move to the next part of the circle

            if (right_x == -1):  # If nothing found, the final value for right x and y is set to the value of x and y
                right_x = x
                right_y = y

            for i in range(0, len(
                    dx_dir_left)):  # Second half of the circle moving from 5pixels down from the x,y pixel and moving left
                left_x = x + dx_dir_left[i]
                left_y = y + dy_dir_left[i]
                if (
                        left_x < 0 or left_y < 0 or left_x >= len_x or left_y >= len_y):  # if we are outside the image's limit, we set left_x and _y to -1 to move to the next position
                    left_x = left_y = -1
                    continue;
                if (skeleton_frames_Pw[k][left_x][
                    left_y] == 255): break;  # if the count is in the last number, it makes left x and left y equals to -1 to move to finish the circle check
                if (i == 13): left_x = left_y = -1

            if (left_x == -1):  # final value for left x and y, we set the value to x and y
                left_x = x
                left_y = y

            # Set the direction of the crack as angle(theta) by using acos formula
            base = right_y - left_y
            height = right_x - left_x
            hypotenuse = math.sqrt(base * base + height * height)

            if (base == 0 and height != 0):
                theta = 90.0
            elif (
                    base == 0 and height == 0):  # If base and height are 0, goes back to check the next pixel in the skeleton
                continue
            else:
                theta = math.degrees(
                    math.acos((base * base + hypotenuse * hypotenuse - height * height) / (2.0 * base * hypotenuse)))

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

                if (theta == 0.0 or theta == 360.0):
                    while (1):
                        pix_y += 1
                        if (pix_y >= len_y):
                            pix_x = x
                            pix_y = y
                            break;
                        if (edges_frames_Pw[k][pix_x][pix_y] == 255): break;

                elif (theta == 90.0):
                    while (1):
                        pix_x -= 1
                        if (pix_x < 0):
                            pix_x = x
                            pix_y = y
                            break;
                        if (edges_frames_Pw[k][pix_x][pix_y] == 255): break;

                elif (theta == 180.0):
                    while (1):
                        pix_y -= 1
                        if (pix_y < 0):
                            pix_x = x
                            pix_y = y
                            break;
                        if (edges_frames_Pw[k][pix_x][pix_y] == 255): break;

                elif (theta == 270.0):
                    while (1):
                        pix_x += 1
                        if (pix_x >= len_x):
                            pix_x = x
                            pix_y = y
                            break;
                        if (edges_frames_Pw[k][pix_x][pix_y] == 255): break;
                else:
                    a = 1
                    radian = math.radians(theta)
                    while (1):
                        pix_x = x - round(a * math.sin(radian))
                        pix_y = y + round(a * math.cos(radian))
                        if (pix_x < 0 or pix_y < 0 or pix_x >= len_x or pix_y >= len_y):
                            pix_x = x
                            pix_y = y
                            break;
                        if (edges_frames_Pw[k][pix_x][pix_y] == 255): break;

                        if (theta > 0 and theta < 90):
                            if (pix_y + 1 < len_y and edges_frames_Pw[k][pix_x][pix_y + 1] == 255):
                                pix_y += 1
                                break;
                            if (pix_x - 1 >= 0 and edges_frames_Pw[k][pix_x - 1][pix_y] == 255):
                                pix_x -= 1
                                break;

                        elif (theta > 90 and theta < 180):
                            if (pix_y - 1 >= 0 and edges_frames_Pw[k][pix_x][pix_y - 1] == 255):
                                pix_y -= 1
                                break;
                            if (pix_x - 1 >= 0 and edges_frames_Pw[k][pix_x - 1][pix_y] == 255):
                                pix_x -= 1
                                break;

                        elif (theta > 180 and theta < 270):
                            if (pix_y - 1 >= 0 and edges_frames_Pw[k][pix_x][pix_y - 1] == 255):
                                pix_y -= 1
                                break;
                            if (pix_x + 1 < len_x and edges_frames_Pw[k][pix_x + 1][pix_y] == 255):
                                pix_x += 1
                                break;

                        elif (theta > 270 and theta < 360):
                            if (pix_y + 1 < len_y and edges_frames_Pw[k][pix_x][pix_y + 1] == 255):
                                pix_y += 1
                                break;
                            if (pix_x + 1 < len_x and edges_frames_Pw[k][pix_x + 1][pix_y] == 255):
                                pix_x += 1
                                break;
                        a += 1

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

    crack_width_list.sort(reverse=True)

    # Convert into real width.
    print(len(crack_width_list))
    if (len(crack_width_list) == 0):
        save_result.append(0)
        real_width = 0
    elif (len(crack_width_list) < 10):
        real_width = round(crack_width_list[len(crack_width_list) - 1] * 0.92, 2) # the value for the pixel in real life is set, this has to be changed to a variable
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

# Save those information into text files.
f1 = open("C:\\Users\\juanc\\OneDrive - KTH\\Python\\Prueba\\width.txt", 'w')
f2 = open("C:\\Users\\juanc\\OneDrive - KTH\\Python\\Prueba\\risk.txt", 'w')

for z in range(0, len(save_result)):
    f1.write(str(save_result[z]) + 'mm' + '\n')
f1.close()

for z in range(0, len(save_risk)):
    f2.write(str(save_risk[z]) + '\n')
f2.close()
