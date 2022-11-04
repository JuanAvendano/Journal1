"""
Created on Wen september 04 2022
@author: jcac

sliding window: https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
balanced histogram: https://theailearner.com/2019/07/19/balanced-histogram-thresholding/

Liste of functions needed for the study of cracks and their properties such as width
"""
import os
import cv2
import math
import numpy as np
from PIL import Image
from numpy import empty
from skimage.util import invert
import matplotlib.pyplot as plt
from skimage.morphology import label
from skimage.morphology import skeletonize
from skimage.morphology import remove_small_objects


def balanced_hist_thresholding(b):
    """
    Obtains the threshold for the image or window studied doing a balance of the values from the histogram of the image
       balanced histogram: https://theailearner.com/2019/07/19/balanced-histogram-thresholding/

    Parameters
    ----------
    :param b: pixel intensity histogram of the image

    Returns
    -------
    i_m : threshold obtained.

    """

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
    while i_s != i_e:
        # If right side is heavier
        if w_r > w_l:
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

def cleanimage(image,minsize):
    """Takes an image and removes  objects having an area equals to minsize. It returns the cleaned image after removing
        the small objects.
    :param image: image to clean
    :param minsize: area of the objects to be removed
    :return: the cleaned Image object
    """
    size=minsize
    regions0=label(image,connectivity=2)
    no_smllobj = remove_small_objects(regions0, min_size=size)  # remove objects with dimensions smaller than 3 pixels
    regions = label(no_smllobj, connectivity=2) # creates regions again after having small objects removed
    resultImage = (regions > 0.5) * 255  # We create the binary image

    return resultImage

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
    skeleton_coord = []

    for i in range(0, len(cropped_frames)):
        img_Pw = (image)
        skeleton_Img = skeletonize(img_Pw)  # generates the skeleton of the image

        skeleton_Img.dtype = 'uint8'  # transform into uint8 data type being numbers from 0 to 255

        skeleton_Img *= 255  # multiply by 255
        """
        ESP:Cuando hay pixeles solos o grupos de pixeles que tienen una longitud menor a 5 pixeles, el algoritmo los empieza
        a estudiar pero como tienen menos de 5 pixeles, al hacer el circulo no encuentra otro pixel del esqueleto a partir
        del cual calcular un angulo entonces determina esa parte como vacia lo que resulta en que el vector de coordenadas
        y el vector de widths termina siendo de diferente tamaño.  Tocaria hacer regiones y quitar las 
        regiones que tienen menos de 5 pixeles peeero esos pixeles pueden en realidad pertenecer a una grieta de otra parte
        en donde si conectan y hacen una linea larga y continua, es decir ocurrir que la ventana corte una grieta justo 
        en un punto donde hay menos de 5 pixeles. Por esto para la prueba de ver las distancias calculadas
        se quitan 3 pixeles que estan solos o que son grupos de pixeles muy cortos:
        """

        skeleton_frames_list.append(skeleton_Img)  # The list which saves the images after the skeleton is obtained.
        for i in range(0, skeleton_Img.shape[0]):
            for j in range(0, skeleton_Img.shape[1]):
                if skeleton_Img[i][j] == 255:
                    skeleton_coord.append([i, j])

    # # ================================================================================================================
    # 2. Detect the edges of the crack.
    # # ================================================================================================================

    # Edges are found using Laplacian. The edges are the negative values and for the sake of the rest of the code
    # they are turned into positive values

    edges_frames_list = []
    imim = (image == 1)
    for i in range(0, len(cropped_frames)):
        imim.dtype = 'uint8'
        thrlap = cv2.Laplacian(imim, cv2.CV_64F)  # image obtained after applying Laplacian for edge detection
        for n in range(0, widthIM):
            for m in range(0, heightIM):
                if thrlap[n, m] > 0:
                    thrlap[n, m] = 255
                else:
                    thrlap[n, m] = 0

        # The list which saves the images after edge detection.
        edges_Img = thrlap
        edges_frames_list.append(thrlap)

    # # ================================================================================================================
    # 3. Calculate the width of the crack.
    # # ================================================================================================================

    # 3.1) Set the direction of the crack by searching skeleton pixels which are 5 pixels away from the skeleton pixel.
    # 3.2) Draw a perpendicular line of the direction
    # 3.3) The perpendicular line meets the edge. The distance is calculated by counting pixels on the line.
    # 3.4) Convert the number of pixels into real mm width, and classify the danger group.

    crack_width_list = []  # Vector fr the elements of the crack width
    listcomplt = empty([len(skeleton_coord), 5])

    # # ===============================================
    # 3.1 Circle
    # # ===============================================

    # dx dir right/left are the pixels in a circle(of 5 pixels radius) around the central pixel that is being studied
    dx_dir_right = [-5, -5, -5, -5, -4, -4, - 3, -3, -2, -1, 0, 1, 2, 3, 3, 4, 4, 5, 5, 5]
    dy_dir_right = [0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 4, 4, 3, 3, 2, 1]

    dx_dir_left = [5, 5, 5, 5, 4, 4, 3, 3, 2, 1, 0, -1, -2, -3, -3, -4, -4, -5, -5, -5]
    dy_dir_left = [0, -1, -2, -3, -3, -4, -4, -5, -5, -5, -5, -5, -5, -5, -4, -4, -3, -3, -2, -1]

    plt.figure('skl+edges', figsize=(10, 10))
    plt.subplot(121)
    plt.imshow(skeleton_Img)
    plt.subplot(122)
    plt.imshow(edges_Img)

    for k in range(0, len(skeleton_coord)):

        x = skeleton_coord[k][0]
        y = skeleton_coord[k][1]

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
        """
        ESP:Para el calculo del width, la distancia es tomada desde el centro de los pixeles entonces en el caso
        de un pixel de esqueleto que tiene el borde justo encima y justo debajo (donde la distancia deberia ser 1,
        solo el pixel de esqueleto) toma medio pixel del borde superior y medio del inferior dando como resultado 
        una distancia (y por ende un width)  de 2. Se pensaria que seria tan simple como quitarle 1 al resultado 
        obtenido pero no porque en los casos en que la grieta no va horizontal o vertical sino que hay un angulo,
        quitar 1 al resultado genera una medicion incorrecta. Aun no me es muy claro que hacer en este caso.


        """
        # The list which saves the width of the crack in terms of pixels
        crack_width_list.append(dist)
        # creates a vector with x,y coord of the skeleton and the corresponding width for that skeleton pixel

        listcomplt[k][0] = x    # X coord of the skeleton pixel
        listcomplt[k][1] = y    # Y coord of the skeleton pixel
        listcomplt[k][2] = dist # Width in pixels
        listcomplt[k][3] = dist * pixel_width   # Width in mm
        listcomplt[k][4] = 0    # Space in case danger group is used



    return crack_width_list, skeleton_coord, skeleton_Img, edges_Img, listcomplt

def danger_group(listcomplt):

    for i in range(0,len(listcomplt)):
        if (listcomplt[i][3] >= 0.3):
            listcomplt[i][4]='high'
            print('risk group : high\n')
        elif (listcomplt[i][3] < 0.3 and listcomplt[i][3] >= 0.2):
            listcomplt[i][4]='medium'
            print('Risk group: medium\n')
        else:
            listcomplt[i][4]='low'
            print('Risk group: low\n')

def imgSaving(path, name, element):
    """
        Saves an obtained image into a specified directory

    Parameters
    ----------
    :param path: path where the image has to be saved
    :param name: name of the resulting file
    :param element: element to be saved

    Returns
    -------

    """
    name = name + '.jpg'
    cv2.imwrite(os.path.join(path, name), element)  # saves the obtained image showing as a .jpg in the folder

def joinwindows(img, windows, i, winH, winW):
    """
       Joins the different windows of an image where cracks have been detected and generates a result image where the
       the thresholded images can be seen together

    Parameters
    ----------
    :param img: Grayscale image
    :param windows: List of windows that have cracks
    :param i: counter from the list of images to check
    :param winH: Height of the window to study
    :param winW: Width of the window to study

    Returns
    -------
    resultImg : Image with the windows joined.
    window : Window that has been studied

    """

    resultImg = img.copy() * 0
    clone = img.copy()

    for j in range(0, len(windows[i])):
        x = windows[i][j][0]  # x coordinate of the upper left corner of the window to evaluate
        y = windows[i][j][1]  # y coordinate of the upper left corner of the window to evaluate
        window = img[y:y + winH, x:x + winW]  # window to evaluate from x and y coord
        red = np.histogram(window.ravel(), bins=256, range=[0, 256])  # hist for the window
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)  # rectangle to see where we are in the image

        trhs = balanced_hist_thresholding(red)
        ret, thresh = cv2.threshold(window, trhs, 255, cv2.THRESH_BINARY)
        xx = 0
        yy = 0
        for k in range(x, x + window.shape[1]):

            for l in range(y, y + window.shape[0]):
                resultImg[l, k] = thresh[xx, yy]
                xx += 1
            yy += 1
            xx = 0

    return resultImg, window

def merge_images(imlist, th, tw):
    """
        Merge two images into one, displayed side by side

    Parameters
    ----------
    :param imlist: List of sliced pictures to stich together
    :param th: Height of result image
    :param tw: Width of result image

    Returns
    -------
    :return: the merged Image object

    """
    result = Image.new('RGB', (tw, th))
    CW = 0  # Counter for the width
    CH = 0  # Counter for the height
    for x in range(len(imlist)):
        imx = Image.open(imlist[x])
        (wx, hx) = imx.size

        if CW < tw:
            result.paste(im=imx, box=(CW, CH))
            CW = CW + wx
        else:
            CW = 0
            CH = CH + hx
            result.paste(im=imx, box=(CW, CH))
            CW = CW + wx

    return result

def selectimg(crack):
    """
       Select the image to study from a list of images

    Parameters
    ----------
    :param crack:
    :param  i:

    Returns
    -------
    image : Image that has been studied.
    img : Studied image in grayscale

    """
    image = cv2.imread(crack + '.jpg')
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image, img

def sliding_window(image, stepSize, windowSize):
    """
    Sliding window within the loaded image. The window corresponds to a squared window that slides across the image
    according to the step size.

     Parameters
    ----------
    image : ndarray
            Image where the window will slide.
    StepSize :  Stride for the window
    windowSize : Size of the squared window

    """

    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])












