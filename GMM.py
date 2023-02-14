"""
Created on january 30 2023
@author: jca
Code to use Gaussian Mixture Models (GMM) to fit two Gaussian distributions on the data for each window.

"""

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from scipy.stats import norm
from skimage.morphology import skeletonize
from skimage.util import invert
import math
from numpy import empty
from Dictionary import sliding_window
import seaborn as sns
from scipy.stats import chisquare, normaltest
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt


# path = r'C:\Users\juanc\Desktop\prueba'
# os.chdir(path)  # Access the path
# image = cv2.imread('_DCS6932_195.jpg')

path = r'C:\Users\juanc\OneDrive - KTH\Python\Prueba\02.1-Cracked_predicted'
os.chdir(path)  # Access the path
image = cv2.imread('_DCS6695_412.jpg')

pixel_width = 0.1

img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
x = 56
y = 56
winW = 28
winH = 28
# window = img[y:(y + winW), x:(x + winH)]

for (x, y, window) in sliding_window(img, stepSize=28, windowSize=(winW, winH)):
    data=window.ravel()
    red = np.histogram(window.ravel(), bins=256, range=[0, 256])
    clone = img.copy()
    # Creates rectangle in the coordinates over the clone to see it in the ploted image
    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)

    # Fit a GMM with two components to the data
    gmm = GaussianMixture(n_components=2)
    gmm.fit(data.reshape(-1, 1))

    # Extract the means and standard deviations of the two Gaussian components
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()

    #cutting point is 1 or 2 stds from each mean, from the left one it is adding and on the right it is subtracting.
    LowBound1 = means[0]-stds[0]
    LowBound2 = means[0] - 2*stds[0]

    UpBound1 = means[1]+stds[1]
    UpBound2 = means[1] + 2*stds[1]

    # Plot the image with the window, the pixels in the window, histogram of the window
    plt.figure('window hist trsh', figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(clone, cmap='gray')
    # plt.subplot(2, 2, 2)
    # plt.imshow(window, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.hist(window.ravel(), 256, [0, 256])
    plt.subplot(2, 2, 3)
    # Plot the data and the fitted GMM
    plt.hist(data, bins=50, density=True)
    a = np.linspace(data.min(), data.max(), (data.max()-data.min()))
    for mean, std in zip(means, stds):
        plt.plot(a, gmm.weights_[0]*np.exp(-(a-mean)**2/(2*std**2))/(std*np.sqrt(2*np.pi)))
    plt.plot(LowBound1)
    plt.subplot(2, 2, 4)
    sns.distplot(window.ravel(), fit=norm, kde=False,label="Density", norm_hist=False  )
    plt.show()
