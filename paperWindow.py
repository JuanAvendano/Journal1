""""
Trying to follow the procedure from my reference paper
"""


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
winW = 28
winH = 28


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(84, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


#Mean image
mn_image = np.zeros((winW,winH))
#Std Dev image
std_image = np.zeros((winW,winH))
#Variance image
var_image = np.zeros((winW,winH))


# For each pixel, count the number of positive
# and negative pixels in the neighborhood
for (x, y, window) in sliding_window(img, stepSize=28, windowSize=(winW, winH)):
    clone = img.copy()
    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    for i in range(1, window.shape[0] - 1):
        for j in range(1, window.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            # neighbour array corresponds to surrounding pixels of the given pixel
            neighbour = [window[i + 1, j - 1], window[i + 1, j], window[i + 1, j + 1], window[i, j - 1], window[i, j + 1],
                         window[i - 1, j - 1], window[i - 1, j], window[i - 1, j + 1]]
            # Mean of the neighbour array
            mn = np.mean(neighbour)
            # Standard deviation of the neighbour array
            std = np.std(neighbour)
            # Variance of the neighbour array
            var = np.var(neighbour)

            mn_image[i, j] = mn
            std_image[i, j] = std
            var_image[i, j] = var


    # Normalize and change datatype to 'uint8' (optional)
    mn_norm = mn_image / mn_image.max() * 255
    mn_image = np.uint8(mn_norm)
    # Normalize and change datatype to 'uint8' (optional)
    std_norm = std_image / std_image.max() * 255
    std_image = np.uint8(std_norm)
    # Normalize and change datatype to 'uint8' (optional)
    var_norm = var_image / var_image.max() * 255
    var_image = np.uint8(var_norm)

    plt.figure('Img-Mean-Std_Dev-Var', figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(clone, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(mn_image, cmap='gray')
    plt.subplot(2, 2, 3)
    # plt.hist(laplacian.ravel(), 256, [-256, 256])
    plt.imshow(std_image, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(var_image, cmap='gray')
    plt.show(block=True)