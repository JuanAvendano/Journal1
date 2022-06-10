import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import face
from PIL import Image, ImageDraw
import os
import cv2

# Load an image
path = r'C:\Users\juanc\OneDrive - KTH\Python\Prueba\02.1-Cracked_predicted'
os.chdir(path)  # Access the path
image = cv2.imread('_DCS6695_412.jpg')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
winW=28
winH=28

def balanced_hist_thresholding(b):
    # Starting point of histogram
    i_s = np.min(np.where(b[0]>0))
    # End point of histogram
    i_e = np.max(np.where(b[0]>0))
    # Center of histogram
    i_m = (i_s + i_e)//2
    # Left side weight
    w_l = np.sum(b[0][0:i_m+1])
    # Right side weight
    w_r = np.sum(b[0][i_m+1:i_e+1])
    # Until starting point not equal to endpoint
    while (i_s != i_e):
        # If right side is heavier
        if (w_r > w_l):
            # Remove the end weight
            w_r -= b[0][i_e]
            i_e -= 1
            # Adjust the center position and recompute the weights
            if ((i_s+i_e)//2) < i_m:
                w_l -= b[0][i_m]
                w_r += b[0][i_m]
                i_m -= 1
        else:
            # If left side is heavier, remove the starting weight
            w_l -= b[0][i_s]
            i_s += 1
            # Adjust the center position and recompute the weights
            if ((i_s+i_e)//2) >= i_m:
                w_l += b[0][i_m+1]
                w_r -= b[0][i_m+1]
                i_m += 1
    return i_m

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(84, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])



for (x, y, window) in sliding_window(img, stepSize=28, windowSize=(winW, winH)):
    red = np.histogram(window.ravel(), bins=256, range=[0, 256])
    clone = img.copy()
    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    # cv2.imshow("Window", clone)
    trhs=balanced_hist_thresholding(red)
    plt.figure('imggdfgw', figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(clone, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(window, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.hist(window.ravel(), 256, [0, 256])





    # Calculate the histogram
    hist = plt.hist(window.ravel(), 256, [0, 256])
    # Total pixels in the image
    total = np.sum(hist[0])
    # calculate the initial weights and the means
    left, right = np.hsplit(hist[0], [0])
    left_bins, right_bins = np.hsplit(hist[1], [0])
    # left weights
    w_0 = 0.0
    # Right weights
    w_1 = np.sum(right) / total
    # Left mean
    mean_0 = 0.0
    weighted_sum_0 = 0.0
    # Right mean
    weighted_sum_1 = np.dot(right, right_bins[:-1])
    mean_1 = weighted_sum_1 / np.sum(right)

    def recursive_otsu1(hist, w_0=w_0, w_1=w_1, weighted_sum_0=weighted_sum_0, weighted_sum_1=weighted_sum_1, thres=trhs,
                        fn_max=-np.inf, thresh=0, total=total):
        if thres <= 255:
            # To pass the division by zero warning
            if np.sum(hist[0][:thres + 1]) != 0 and np.sum(hist[0][thres + 1:]) != 0:
                # Update the weights
                w_0 += hist[0][thres] / total
                w_1 -= hist[0][thres] / total
                # Update the mean
                weighted_sum_0 += (hist[0][thres] * hist[1][thres])
                mean_0 = weighted_sum_0 / np.sum(hist[0][:thres + 1])
                weighted_sum_1 -= (hist[0][thres] * hist[1][thres])
                if thres == 255:
                    mean_1 = 0.0
                else:
                    mean_1 = weighted_sum_1 / np.sum(hist[0][thres + 1:])
                # Calculate the between-class variance
                out = w_0 * w_1 * ((mean_0 - mean_1) ** 2)
                # # if variance maximum, update it
                if out > fn_max:
                    fn_max = out
                    thresh = thres
            return recursive_otsu1(hist, w_0=w_0, w_1=w_1, weighted_sum_0=weighted_sum_0, weighted_sum_1=weighted_sum_1,
                                   thres=thres + 1, fn_max=fn_max, thresh=thresh, total=total)
        # Stopping condition
        else:
            return fn_max, thresh




    # Check the results
    var_value, thresh_value = recursive_otsu1(hist, w_0=w_0, w_1=w_1, weighted_sum_0=weighted_sum_0,
                                              weighted_sum_1=weighted_sum_1, thres=1, fn_max=-np.inf, thresh=0, total=total)
    print(var_value, thresh_value)