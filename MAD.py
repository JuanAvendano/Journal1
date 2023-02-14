"""
Created on february 10 2023
@author: jca
Code to use Median Absolute Deviation.

"""
import numpy as np
import cv2
import os
from scipy.stats import norm
from skimage.morphology import skeletonize
from skimage.util import invert
import math
from numpy import empty
from Dictionary import sliding_window
import seaborn as sns
from scipy.stats import chisquare, normaltest
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def detect_outliers_mad(data, threshold=2):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    lower_bound = median - threshold * mad
    upper_bound = median + threshold * mad
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    medianlist = [x for x in data if x == median ]
    inliers = [x for x in data if lower_bound <= x and upper_bound >= x]
    return outliers, median, medianlist, inliers


path = r'C:\Users\juanc\OneDrive - KTH\Python\Prueba\02.1-Cracked_predicted'
os.chdir(path)  # Access the path
image = cv2.imread('_DCS6695_412.jpg')

img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
winW = 28
winH = 28



for (x, y, window) in sliding_window(img, stepSize=28, windowSize=(winW, winH)):
    WinData = window.ravel()
    red = np.histogram(WinData, bins=256, range=[0, 256])
    clone = img.copy()
    # Creates rectangle in the coordinates over the clone to see it in the ploted image
    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)

    # Detect Outliers using MAD
    Outl, med, medlist, inliers = detect_outliers_mad(WinData)
    if x==112 and y==28:
        # Plot the image with the window, the pixels in the window, histogram of the window
        plt.figure('window hist trsh', figsize=(19, 10))
        plt.subplot(2, 3, 1)
        plt.imshow(clone, cmap='gray')
        plt.title('Sub-image')

        plt.subplot(2, 3, 2)
        plt.hist(WinData, 256, [0, 256])
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Intensity histogram')

        plt.subplot(2, 3, 3)
        sns.distplot(WinData, fit=norm, kde=True, label="Density", norm_hist=False)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Normal fitting')

        plt.subplot(2, 3, 4)
        bins=WinData.max()-WinData.min()
        plt.hist(WinData, bins=bins,  density=True)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Intensity histogram')

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

        plt.show()

        # Perform Chi-square test for normality
        stat_chi2, p_chi2 = chisquare(inliers)
        print("Chi-square test statistic: {:.3f}".format(stat_chi2))
        print("p-value: {:.3f}".format(p_chi2))

        # Perform the normaltest test
        stat_norm, p_norm = normaltest(inliers)
        print("Normal test statistic: {:.3f}".format(stat_norm))
        print("p-value: {:.3f}".format(p_norm))

        # Plot for fitting a normal distributin for the InLiers
        plt.figure('Inliers', figsize=(19, 10))
        sns.distplot(inliers, fit=norm, kde=False, label="Density", norm_hist=False)
        plt.text(0.55, 0.4, 'Chi-square test statistic: {:.3f}\np-value: {:.3f}'.format(stat_chi2, p_chi2),
                 fontsize=9, transform=plt.gcf().transFigure)
        plt.text(0.55, 0.35, 'Normal test statistic: {:.3f}\np-value: {:.3f}'.format(stat_norm, p_norm),
                 fontsize=9, transform=plt.gcf().transFigure)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Normal fitting for inliers')
        plt.show()



