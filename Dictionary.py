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

    # plt.figure('skl+edges', figsize=(10, 10))
    # plt.subplot(121)
    # plt.imshow(skeleton_Img)
    # plt.subplot(122)
    # plt.imshow(edges_Img)

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
        listcomplt[k][2] = max(dist-1,0) # Width in pixels
        listcomplt[k][3] = max(dist-1,0) * pixel_width   # Width in mm
        listcomplt[k][4] = 0    # Space in case danger group is used



    return crack_width_list, skeleton_coord, skeleton_Img, edges_Img, listcomplt

def danger_group(listcomplt):
    """
          Defines a danger group for a certain crack depending on the width. The limits correspond to high if the crack
          width is bigger than 0.3mm, medium if it is between 0.2 and 0.3 and low if it is less tha 0.2mm.
          Changes the lisst 5th column to the corresponding danger group.

       Parameters
       ----------
       :param listcomplt: List of cracks with their corresponding widths. The widths must be in the fourth column

       Returns
       -------

       """
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

def detect_outliers_mad(data, threshold):
    """
       Takes the image to analyze as a 1D array and uses Median Absolute Deviation to determine the outliers using
       a threshold that describes a distance from the median of the array. The usual value for the threshold is 3.5
       since it corresponds roughly to the 99.7th percentile of the standard normal distribution.

    Parameters
    ----------
    :param data: 1D array of intensity values
    :param threshold: Threshold for MAD method

    Returns
    -------
    outliers : List of elements considered as outliers.
    median : Median of the data
    medianlist: Elements from the data being the median
    inliers: List of elements considered as inliers
    lower_bound: Lower cutting point to divide inliers and outliers

    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    lower_bound = median - threshold * mad
    upper_bound = median + threshold * mad
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    medianlist = [x for x in data if x == median ]
    inliers = [x for x in data if lower_bound <= x and upper_bound >= x]
    return outliers, median, medianlist, inliers, lower_bound

def final_subimage(image, imageColor,threshold,pixel_width):
    """
    Applies MAD with the introduced threshold for the input image.
    :param image: grayscale image
    :param threshold: threshold for MAD
    :param pixel_width: pixel width in mm
    :return:resultImg:binary image
            finalsubimg: final sub image
            low_bound: threshold given by MAD
            medlist: median
            Outl: Outliers list
    """
    # Transform to 1D array
    WinData = image.ravel()

    # Apply MAD procedure to detect outliers
    Outl, med, medlist, inliers, low_bound = detect_outliers_mad(WinData, threshold)

    # Creates the result image
    resultImg = image.copy() * 0

    # Creates thresh, binary image obtained using the lowbound threshold obtained previously with MAD
    ret, thresh = cv2.threshold(image, low_bound, 255, cv2.THRESH_BINARY)
    thresh = invert(thresh)

    # Takes the image with only the crack and removes small objects according to the specified size
    resultImage = cleanimage(thresh,3)

    # Get the widths, coordinate of the skeleton, img with skeleton,img with edges, info list
    widths, coordsk, skframes, edgesframes, completeList = CrackWidth(resultImage // 255, pixel_width)

    # Sets the image width and heigh
    width = image.shape[0]
    height = image.shape[1]

    # Creates the finalsubimage element that will have the crack highlighted and the rest of the pixel in grayscale
    finalsubimg = empty([width, height, 3], dtype=np.uint8)
    for x in range(0, width):
        for y in range(0, height):
            if skframes[x, y] > 0:
                finalsubimg[x, y] = [255, 0, 0]  # If pixel is part of skeleton paint red
            elif resultImage[x, y] > 0:
                finalsubimg[x, y] = [255, 255, 0]  # If pixel is part of crack paint yellow
            elif edgesframes[x, y] > 0:
                finalsubimg[x, y] = [0, 0, 255]  # If pixel is part of edge, paint blue
            else:
                finalsubimg[x, y] = imageColor[x, y]

    return resultImg, finalsubimg, low_bound, medlist, Outl

def imgSaving(path, name, element):
    """
        Saves an obtained image into a specified directory

    Parameters
    ----------
    :param path: path where the image has to be saved
    :param name: name of the resulting file
    :param element: Numpy element to be saved as an Image

    Returns
    -------

    """
    name = os.path.join(path, name+'.png')
    image_element=Image.fromarray(element)  # Transforms the NumPy array into an image element
    image_element.save(name)  # saves the obtained image showing as a .png in the folder

def BinarySaving(path, name, element):
    """
        Saves an obtained image into a specified directory

    Parameters
    ----------
    :param path: path where the image has to be saved
    :param name: name of the resulting file
    :param element: Numpy element to be saved as an Image

    Returns
    -------

    """
    name = name + '.png'
    cv2.imwrite(os.path.join(path, name), element)  # saves the obtained image showing as a .png in the folder

def instersection_gaussians(gmm, i, j):
    """
    Find the intersection of two Gaussian distributions fitted with a Gaussian mixture model

    Parameters:
        gmm : GaussianMixture object
            Fitted Gaussian mixture model
        i : int
            Index of the first Gaussian component
        j : int
            Index of the second Gaussian component

    Returns:
        x_intersection : float
            The x value at the intersection point
    """
    mu_i, cov_i = gmm.means_[i], gmm.covariances_[i]
    mu_j, cov_j = gmm.means_[j], gmm.covariances_[j]
    sigma_i = np.sqrt(cov_i)
    sigma_j = np.sqrt(cov_j)
    pi_i = gmm.weights_[i]
    pi_j = gmm.weights_[j]


    def f(x):
        return pi_i * np.exp(-(x - mu_i) ** 2 / (2 * sigma_i ** 2)) - pi_j * np.exp(-(x - mu_j) ** 2 / (2 * sigma_j ** 2))


    x_intersection = root_scalar(f, bracket=[mu_i - 3 * sigma_i, mu_j + 3 * sigma_j]).root
    return x_intersection

def joinwindows(img, windows, i, winH, winW,threshold):
    """
       Takes windows (list of windows) from a subimage (can be multiple subimages in a list), apply a threshold method
       and puts all the windows results together in their corresponding place in the subimage. The result is a binary
       image, the results will be added to the original image after calculating widths lenghts and others.

    Parameters
    ----------
    :param img: Grayscale image
    :param windows: List of windows that have cracks
    :param i: counter from the list of images to check
    :param winH: Height of the window to study
    :param winW: Width of the window to study
    :param threshold: Threshold for MAD method, 0 if Balanced threshold is selected

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
        WinData = window.ravel()
        red = np.histogram(window.ravel(), bins=256, range=[0, 256])  # hist for the window
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)  # rectangle to see where we are in the image

        if threshold==0:
            trhs = balanced_hist_thresholding(red)
        else :
            Outl, med, medlist, inliers, trhs = detect_outliers_mad(WinData,threshold)


        ret, thresh = cv2.threshold(window, trhs, 255, cv2.THRESH_BINARY)
        thresh=invert(thresh)
        # Puts the different binary windows(28x28, usually) obtained after threshold into a binary subimage (224x224)
        xx = 0
        yy = 0
        for k in range(x, x + window.shape[1]):

            for l in range(y, y + window.shape[0]):
                resultImg[l, k] = thresh[xx, yy]
                xx += 1
            yy += 1
            xx = 0

    return resultImg, window

def merge_images(imlist, nr, nc):
    """
        Merge subimages into one, displayed according to the number of columns and rows given as input.

    Parameters
    ----------
    :param imlist: List of sliced pictures to stich together. All the images must be same size
    :param nr: number of rows of result image
    :param nc: number of columns of result image

    Returns
    -------
    :return: the merged Image object

    """
    images = [Image.open(path) for path in imlist]

    # Get dimensions of result image
    widths, heights = zip(*(i.size for i in images))
    result_width = max(widths) * nc
    result_height = max(heights) * nr

    # Create new Image
    result = Image.new('RGB', (result_width, result_height))

    # Paste subimages into result image
    for row in range(nr):
        for col in range(nc):
            index = row * nc + col
            if index >= len(images):
                break
            image = images[index]
            x_offset = col * max(widths)
            y_offset = row * max(heights)
            result.paste(image, (x_offset, y_offset))

    return result

def merge_images_with_labels(imlist, nr, nc):
    """
        Merge subimages into one with divisions for the subimages used and the name of the subimages over each
        subimage. The name corresponds to the last part of the subimage file name.
        The final image is displayed according to the number of columns and rows given as input

    Parameters
    ----------
    :param imlist: List of sliced pictures to stich together. All the images must be same size
    :param nr: number of rows of result image
    :param nc: number of columns of result image

    Returns
    -------
    :return: the merged Image object

    """
    images = [Image.open(path) for path in imlist]

    # Get dimensions of result image
    widths, heights = zip(*(i.size for i in images))
    result_width = max(widths) * nc
    result_height = max(heights) * nr

    result = Image.new('RGB', (result_width, result_height))    # Create new Image
    # Paste subimages into result image
    for row in range(nr):
        for col in range(nc):
            index = row * nc + col
            if index >= len(images):
                break
            image = images[index]
            x_offset = col * max(widths)
            y_offset = row * max(heights)
            result.paste(image, (x_offset, y_offset))
            # Draw bounding box around image
            draw = ImageDraw.Draw(result)
            draw.rectangle((x_offset, y_offset, x_offset + image.width, y_offset + image.height), outline='white')

            # Add label on top of image
            label = f"{imlist[index].split('_')[-1].split('.')[0]}"
            font = ImageFont.truetype("arial.ttf", 20)
            text_width, text_height = draw.textsize(label, font)
            text_x = x_offset + (image.width - text_width) // 2
            text_y = y_offset + text_height - 5
            draw.rectangle((text_x - 5, text_y - 5, text_x + text_width + 5, text_y + text_height + 5),
                           fill=(0, 0, 0, 128))
            draw.text((text_x, text_y), label, font=font, fill='white')

    return result

def selectimg(crack):
    """
       Select the image to study from a list of images and returns the image in grayscale

    Parameters
    ----------
    :param crack:

    Returns
    -------
    image : Image that has been studied.
    img : Studied image in grayscale

    """
    image = cv2.imread(crack + '.png')
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

# # ================================================================================================================
# 2. Specific plots
# # ================================================================================================================
def plot_hist_kde_outl(image,clone,Outl,medlist):

    # Transform to 1D array
    WinData = image.ravel()
    # Plot the image with the window, the pixels in the window, histogram of the window
    fig=plt.figure('Subimg Hist Normfit Outliers ', figsize=(19, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Window')

    plt.subplot(2, 3, 2)
    plt.imshow(clone, cmap='gray')
    plt.title('Sub-image region')

    # plt.subplot(2, 3, 2)
    # plt.hist(WinData, 256, [0, 256] )
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Intensity histogram' )


    plt.subplot(2, 3, 3)
    sns.distplot(WinData, fit=norm, kde=True, label="Density", norm_hist=False)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    plt.title('Normal fitting')

    plt.subplot(2, 3, 4)
    bins=WinData.max()-WinData.min()
    plt.hist(WinData, bins=bins,  density=True)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Intensity histogram (zoom)')

    plt.subplot(2, 3, 5)
    plt.hist(Outl, bins=50, label= 'Outliers')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Outliers')

    plt.subplot(2, 3, 6)
    plt.hist(WinData, bins=bins,color='blue' , label= 'DataSet')
    plt.hist(Outl, bins=bins, color='red', label='Outliers')
    plt.hist(medlist, bins=2, color='green', label='Median')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    plt.title('Histogram with Outliers')


    return fig

def plot_hist_eq(image):
    WinData = image.ravel()
    # Histogram and CDF for the initial image
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    #Mask for equalization of image
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    # Equalizaed image
    img2 = cdf[image]
    hist2, bins = np.histogram(img2.flatten(), 256, [0, 256])
    cdf2 = hist2.cumsum()
    cdf_normalized2 = cdf2 * hist2.max() / cdf2.max()


    #Plot the image
    plt.figure('img', figsize=(10, 10))
    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Image')

    #subplot histogram
    plt.subplot(132)
    # plt.plot(cdf_normalized, color='b')
    plt.hist(image.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(( 'histogram'), loc='upper left')

    # plt.subplot(2, 2, 3)
    # plt.imshow(img2, cmap='gray')
    # plt.title('Image2')

    #subplot equalized histogram
    plt.subplot(133)
    # plt.plot(cdf_normalized2, color='b')
    plt.hist(img2.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf2', 'histogram2'), loc='upper left')
    plt.show()

def hist_equalization(image):
    WinData = image.ravel()
    # Histogram and CDF for the initial image
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Mask for equalization of image
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    # Equalizaed image
    img2 = cdf[image]
    hist2, bins = np.histogram(img2.flatten(), 256, [0, 256])
    cdf2 = hist2.cumsum()
    cdf_normalized2 = cdf2 * hist2.max() / cdf2.max()

    # if print1 == True:
    # Plot the image
    plt.figure('img' , figsize=(10, 10))
    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Image')

    # subplot histogram
    plt.subplot(132)
    # plt.plot(cdf_normalized, color='b')
    plt.hist(image.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('histogram'), loc='upper left')

    # subplot equalized histogram
    plt.subplot(133)
    # plt.plot(cdf_normalized2, color='b')
    plt.hist(img2.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf2', 'histogram2'), loc='upper left')
    plt.show()
