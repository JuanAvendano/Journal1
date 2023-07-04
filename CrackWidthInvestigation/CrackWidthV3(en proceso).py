"""
Created on Wen september 04 2022
@author: jcac

Liste of functions needed for the study of cracks and their properties such as width
"""
import os
import cv2
import math
from numpy import empty
import numpy as np
from skimage.morphology import skeletonize

def CrackWidth(image, pixel_width):
    """
       Calculation of the width of a crack

    Parameters
    ----------
    :param image: ndarray Input data where the crack is located
    :param pixel_width:

    Returns
    -------
    list : A list of the different point of the crack with the corresponding width.

    """

    cropped_frames = [image]
    widthIM = image.shape[0]
    heightIM = image.shape[1]
    pixel_width = pixel_width

    # # ================================================================================================================
    # 1. Extract the skeleton of the crack.
    # # ================================================================================================================

    skeleton_frames_list = []
    skeleton_coords = []

    for i in range(0, len(cropped_frames)):
        img_Pw = (image)
        skeleton = skeletonize(img_Pw)*255  # generates the skeleton of the image

        # skeleton_Img.dtype = 'uint8'  # transform into uint8 data type being numbers from 0 to 255
        # skeleton_Img *= 255  # multiply by 255

        skeleton_frames_list.append(skeleton)  # The list which saves the images after the skeleton is obtained.

        skeleton_coords = np.argwhere(skeleton == 255)

        # for i in range(0, skeleton.shape[0]):
        #     for j in range(0, skeleton.shape[1]):
        #         if skeleton[i][j] == 255:
        #             skeleton_coord.append([i, j])

    # # ================================================================================================================
    # 2. Detect the edges of the crack.
    # # ================================================================================================================

    # Edges are found using Laplacian. The edges are the negative values and for the sake of the rest of the code
    # they are turned into positive values

    edges_frames_list = []
    # imim = (image == 1)
    for i in range(0, len(cropped_frames)):
        # imim.dtype = 'uint8'
        thrlap = np.abs(cv2.Laplacian(image.astype('uint8'), cv2.CV_64F)) > 0 # image obtained after applying Laplacian for edge detection

        # for n in range(0, widthIM):
        #     for m in range(0, heightIM):
        #         if thrlap[n, m] > 0:
        #             thrlap[n, m] = 255
        #         else:
        #             thrlap[n, m] = 0

        # The list which saves the images after edge detection.
        edges_Img = thrlap.astype('uint8') * 255
        edges_frames_list.append(thrlap)

    # # ================================================================================================================
    # 3. Calculate the width of the crack.
    # # ================================================================================================================

    # 3.1) Set the direction of the crack by searching skeleton pixels which are 5 pixels away from the skeleton pixel.
    # 3.2) Draw a perpendicular line of the direction
    # 3.3) The perpendicular line meets the edge. The distance is calculated by counting pixels on the line.
    # 3.4) Convert the number of pixels into real mm width, and classify the danger group.

    crack_width_list = []  # Vector for the elements of the crack width
    listcomplt = empty([len(skeleton_coords), 5])

    # # ===============================================
    # 3.1 Circle
    # # ===============================================

    # dx dir right/left are the pixels in a circle(of 5 pixels radius) around the central pixel that is being studied
    dx_dir_right = [-5, -5, -5, -5, -4, -4, - 3, -3, -2, -1, 0, 1, 2, 3, 3, 4, 4, 5, 5, 5]
    dy_dir_right = [0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 4, 4, 3, 3, 2, 1]

    dx_dir_left = [5, 5, 5, 5, 4, 4, 3, 3, 2, 1, 0, -1, -2, -3, -3, -4, -4, -5, -5, -5]
    dy_dir_left = [0, -1, -2, -3, -3, -4, -4, -5, -5, -5, -5, -5, -5, -5, -4, -4, -3, -3, -2, -1]

    for k in range(0, len(skeleton_coords)):

        x = skeleton_coords[k][0]
        y = skeleton_coords[k][1]

        len_x = skeleton_frames_list[0].shape[0]  # generates a value equal to the length of the skeleton image
        len_y = skeleton_frames_list[0].shape[1]  # generates a value equal to the height of the skeleton image

        circle_right_x = circle_right_y = circle_left_x = circle_left_y = -1  # initial value for right and left x y

        # Estimating the direction of the crack from skeleton
        # We start checking the pixel in x,y that has a value of 255. We proceed to check the pixels that also have a
        # 255 value in a radius of 5 pixels.
        for i in range(0, len(dx_dir_right)):  # First half of the circle moving from 5pixels up from the x,y pixel and
            # moving right
            circle_right_x = x + dx_dir_right[i]  # Point in the circle to the right
            circle_right_y = y + dy_dir_right[i]  # Point in the circle to the left
            if circle_right_x < 0 or circle_right_y < 0 or circle_right_x >= len_x or circle_right_y >= len_y:  # if we are outside the image's limit, we set righ_x and _y to -1 to move to the next position
                circle_right_x = circle_right_y = -1
                continue;
            if skeleton_frames_list[0][circle_right_x][circle_right_y] == 255:
                break;  # Check the place we are, if the skeleton image has something (a value of 1)in the place we are checking or if it is 0. If it has a value of 1, breaks (goes out of the for) and the values for circle_right_x and _y are the distance where something was found
            if i == 19:
                circle_right_x = circle_right_y = -1  # if the count is in the last number, it makes right x and right y equals to -1 to move to the next part of the circle

        if circle_right_x == -1:  # If nothing found, the final value for right x and y is set to the value of x and y
            circle_right_x = x
            circle_right_y = y

        for i in range(0, len(
                dx_dir_left)):  # Second half of the circle moving from 5pixels down from the x,y pixel and moving left
            circle_left_x = x + dx_dir_left[i]
            circle_left_y = y + dy_dir_left[i]
            if circle_left_x < 0 or circle_left_y < 0 or circle_left_x >= len_x or circle_left_y >= len_y:  # if we are outside the image's limit, we set circle_left_x and _y to -1 to move to the next position
                circle_left_x = circle_left_y = -1
                continue;
            if skeleton_frames_list[0][circle_left_x][circle_left_y] == 255:
                break;
            if i == 19:
                circle_left_x = circle_left_y = -1  # if the count is in the last number, it makes left x and left y equals to -1 to move to finish the circle check

        if circle_left_x == -1:  # final value for left x and y, we set the value to x and y
            circle_left_x = x
            circle_left_y = y

        # # ===============================================
        # Set the direction of the crack as angle(theta) by using atan formula
        base = circle_right_y - circle_left_y
        height = circle_left_x - circle_right_x  # circle_right_x - circle_left_x # Since in the image the x increases going down, the order needed to be changed

        if base == 0 and height != 0:  # If base is 0 BUT height is not 0, it means it is a vertical line thus angle is 90
            alpha = 90.0
        elif base == 0 and height == 0:  # If base and height are 0, goes back to check the next pixel in the skeleton
            # The value for the width is set as 0 for this case showing that it is a pixel
            # in the skeleton but the lenght of that part is less than the radius stablished.
            # This way the length of the width list is the same as the skeleton pixel
            # coordinates.
            listcomplt[k][0] = x
            listcomplt[k][1] = y
            listcomplt[k][2] = 0
            listcomplt[k][3] = 0
            crack_width_list.append(0)
            continue
        else:
            alpha = math.degrees(math.atan((height / base)))

        # # ===============================================
        # 3.2 Perpendicular line to skeleton direction
        # # ===============================================
        """ Calculate the distance if the perpendicular line meets the edge of the crack."""

        theta = alpha + 90  # add 90 to have the perpendicular angle
        dist = 0

        for i in range(0, 2):
            pix_x = x
            pix_y = y
            if theta > 360:
                theta -= 360
            elif theta < 0:
                theta += 360

            if theta == 0.0 or theta == 360.0:  # if the line is horizontal, checks the edge adding 1 to pix_y
                # until it reaches the edge
                while (1):  # Same as while (True)
                    pix_y += 1
                    if pix_y >= len_y:
                        pix_x = x
                        pix_y = y
                        break;
                    if edges_frames_list[0][pix_x][pix_y] == 255: break;

            elif theta == 90.0:  # if the line is vertical,checks the edge subtracting 1 to pix_x until it reaches the edge
                while (1):
                    pix_x -= 1
                    if pix_x < 0:
                        pix_x = x
                        pix_y = y
                        break;
                    if edges_frames_list[0][pix_x][pix_y] == 255: break;

            elif theta == 180.0:  # if the line is horizontal, checks the edge subtracting 1 to pix_y until it reaches the edge
                while (1):
                    pix_y -= 1
                    if (pix_y < 0):
                        pix_x = x
                        pix_y = y
                        break;
                    if (edges_frames_list[0][pix_x][pix_y] == 255): break;

            elif theta == 270.0:  # if the line is vertical, checks the edge adding 1 to pix_x until it reaches the edge
                while (1):
                    pix_x += 1
                    if pix_x >= len_x:
                        pix_x = x
                        pix_y = y
                        break;
                    if edges_frames_list[0][pix_x][pix_y] == 255: break;

            # # ===============================================
            # If the angle is other then enters to this section:
            # # ===============================================
            else:
                a = 1  # distance in terms of pixels at which to search
                radian = math.radians(theta)
                while 1:
                    pix_x = x - round(a * math.sin(radian))
                    pix_y = y + round(a * math.cos(radian))
                    if pix_x < 0 or pix_y < 0 or pix_x >= len_x or pix_y >= len_y:
                        pix_x = x
                        pix_y = y
                        break;
                    if edges_frames_list[0][pix_x][pix_y] == 255: break;

                    if 0 < theta < 90:
                        if pix_y + 1 < len_y and edges_frames_list[0][pix_x][pix_y + 1] == 255:
                            pix_y += 1
                            break;
                        if pix_x - 1 >= 0 and edges_frames_list[0][pix_x - 1][pix_y] == 255:
                            pix_x -= 1
                            break;

                    elif 90 < theta < 180:
                        if pix_y - 1 >= 0 and edges_frames_list[0][pix_x][pix_y - 1] == 255:
                            pix_y -= 1
                            break;
                        if pix_x - 1 >= 0 and edges_frames_list[0][pix_x - 1][pix_y] == 255:
                            pix_x -= 1
                            break;

                    elif 180 < theta < 270:
                        if pix_y - 1 >= 0 and edges_frames_list[0][pix_x][pix_y - 1] == 255:
                            pix_y -= 1
                            break;
                        if pix_x + 1 < len_x and edges_frames_list[0][pix_x + 1][pix_y] == 255:
                            pix_x += 1
                            break;

                    elif 270 < theta < 360:
                        if pix_y + 1 < len_y and edges_frames_list[0][pix_x][pix_y + 1] == 255:
                            pix_y += 1
                            break;
                        if pix_x + 1 < len_x and edges_frames_list[0][pix_x + 1][pix_y] == 255:
                            pix_x += 1
                            break;
                    a += 1  # If nothing was found with a distance of 1, it increases the distance

            # # ===============================================
            # 3.3 Distance to edge
            # # ===============================================
            # Calculates the distance from the center pixel to the edge pixel found and adds it to the var dist. It then
            # adds 180 degrees to the angle to check the distance in the other direction. Finally it adds it again to the
            # var dist to get the final distance from both sides
            dist += math.sqrt((y - pix_y) ** 2 + (x - pix_x) ** 2)
            theta += 180

        # The list which saves the width of the crack in terms of pixels
        crack_width_list.append(dist)
        # creates a vector with x,y coord of the skeleton and the corresponding width for that skeleton pixel

        listcomplt[k][0] = x    # X coord of the skeleton pixel
        listcomplt[k][1] = y    # Y coord of the skeleton pixel
        listcomplt[k][2] = max(dist-1,0) # Width in pixels
        listcomplt[k][3] = max(dist-1,0) * pixel_width   # Width in mm
        listcomplt[k][4] = 0    # Space in case danger group is used



    return crack_width_list, skeleton_coords, skeleton, edges_Img, listcomplt
