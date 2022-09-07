"""
Created on Wen august 22 2022
@author: jca

sliding window: https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
balanced histogram: https://theailearner.com/2019/07/19/balanced-histogram-thresholding/
Crack width initially based on the GithHub repository: https://github-com.translate.goog/Garamda/Concrete_Crack_Detection_and_Analysis_SW?_x_tr_sl=ko&_x_tr_tl=en&_x_tr_hl=es&_x_tr_pto=wapp


"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import Dictionary as dict
from skimage.util import invert

path = r'C:\Users\juanc\OneDrive - KTH\Journals\01-Quantification\Image_list'


pixel_width=0.08
winW = 28
winH = 28
crack=0

# region crack info
# Crack 1
_DCS6931_078= [[168, 140], [168, 168], [168, 196], [196, 196]]
_DCS6931_111= [[168, 0], [196, 0], [168, 28], [168, 56], [168, 84], [168, 112], [168, 140], [140,140], [140, 168], [112, 168],  [112, 196], [140, 196]]
_DCS6931_144= [[112, 0], [140, 0], [140, 28], [140, 56], [112, 84], [112, 112], [84, 140], [112, 140], [84, 168], [112, 168], [84, 196], [112, 196]]
_DCS6931_177= [[84, 0], [112, 0], [84, 28], [84, 56], [84, 84], [84, 112], [84, 140], [84, 168], [84, 196]]
_DCS6931_210= [[84, 0], [84, 28], [56, 56], [56, 140], [84, 140], [84, 168], [84, 196], [112, 196]]
_DCS6931_243= [[112, 0], [112, 28], [112, 56], [112, 84], [112, 112], [112, 140], [112, 168], [112, 196]]
_DCS6931_276= [[196, 0]]

# Crack 2
_DCS6932_193=[[112, 196], [196, 140], [196, 168], [168, 196], [196, 196]]
_DCS6932_194=[[28, 56], [56, 56], [84, 56], [112, 56], [140, 56], [168, 56], [196, 56], [0, 84], [28, 84], [56, 84], [84, 84], [140, 84], [0, 112]]
_DCS6932_195= [[0, 56], [28, 56], [56, 56], [84, 56], [112, 56], [112, 84], [140, 84], [168, 84], [196, 84], [196, 112]]
_DCS6932_196=[[0, 112], [28, 112], [56, 112], [84, 112]]
_DCS6932_225= [[196, 84], [168, 112], [84, 140], [112, 140], [140, 140], [84, 168], [0, 196], [28, 196], [56, 196]]
_DCS6932_226=[[56, 0], [84, 0], [112, 0], [140, 0], [0, 28], [28, 28], [56, 28], [112, 28], [0, 56]]

# Crack 3
_DCS6928_137= [[28, 0], [0, 28], [56, 28], [56, 56], [84, 56], [112, 84], [112, 112], [140, 112], [140, 140], [140, 168], [140, 196], [168, 196]]
_DCS6928_170= [[168, 0], [168, 28], [168, 56], [168, 84], [168, 112], [196, 140], [196, 168], [196, 196]]
_DCS6928_203= [[168, 0], [168, 28], [168, 56], [168, 84], [168, 112], [168, 140], [168, 168], [168, 196]]
_DCS6928_236= [[196, 0], [196, 28], [168, 56], [196, 56], [168, 84], [196, 84], [196, 112], [196, 140], [196, 168], [196, 196]]
_DCS6928_270= [[0, 0], [0, 28], [0, 56], [0, 84], [0, 112], [0, 140], [0, 168], [28, 168], [28, 196]]
_DCS6928_303= [[0, 0], [28, 0], [0, 28], [28, 28], [0, 56], [0, 84], [0, 112], [0, 140], [0, 168], [0, 196]]
_DCS6928_336= [[0, 0], [0, 28], [0, 56], [0, 84], [28, 112], [0, 140], [28, 140], [28, 168], [0, 196]]
_DCS6928_369= [[0, 0], [0, 28], [0, 56], [0, 84], [0, 112], [0, 140]]

# Crack 4
_DCS6928_447= [[168, 168], [196, 168], [140, 196], [168, 196]]
_DCS6928_448= [[0, 140], [28, 140], [0, 168], [28, 168], [56, 168], [84, 168], [112, 168], [140, 168], [168, 168], [196, 168]]
_DCS6928_449= [[0, 168], [0, 196], [28, 196]]
_DCS6928_482= [[56, 0], [84, 0], [84, 28], [112, 56], [112, 84], [140, 84], [140, 112], [168, 112]]
_DCS6928_483= [[0, 140], [28, 140], [140, 140], [168, 140], [196, 140], [28, 168], [56, 168], [84, 168], [112, 168]]
_DCS6928_484= [[0, 112], [28, 112], [56, 112], [0, 140], [84, 140], [112, 140], [140, 168], [168, 168], [196, 168]]
_DCS6928_485= [[0, 168], [0, 196], [28, 196], [56, 196], [84, 196], [112, 196], [140, 196]]
_DCS6928_486= [[56, 140], [84, 140], [112, 140], [140, 140], [168, 140], [196, 140], [28, 168], [56, 168], [0, 196], [28, 196]]
_DCS6928_487= [[0, 112], [28, 112], [56, 112], [84, 112], [112, 112], [140, 112], [168, 112], [196, 112], [0, 140]]
_DCS6928_488= [[0, 112], [28, 112], [28, 140]]
_DCS6928_518= [[140, 0], [168, 0], [196, 0]]


Crack1=['_DCS6931_078','_DCS6931_111','_DCS6931_144','_DCS6931_177','_DCS6931_210','_DCS6931_243','_DCS6931_276']
Crack2=[ '_DCS6932_193' ,'_DCS6932_194','_DCS6932_195','_DCS6932_196','_DCS6932_225','_DCS6932_226']
Crack3=[ '_DCS6928_137' ,'_DCS6928_170','_DCS6928_203','_DCS6928_236','_DCS6928_270','_DCS6928_303','_DCS6928_336','_DCS6928_369']
Crack4=[ '_DCS6928_447' ,'_DCS6928_448','_DCS6928_449','_DCS6928_482','_DCS6928_483','_DCS6928_484','_DCS6928_485','_DCS6928_486','_DCS6928_487','_DCS6928_488','_DCS6928_518']

WindowsCrack1=[_DCS6931_078, _DCS6931_111, _DCS6931_144, _DCS6931_177, _DCS6931_210, _DCS6931_243, _DCS6931_276]
WindowsCrack2=[ _DCS6932_193 ,_DCS6932_194,_DCS6932_195,_DCS6932_196,_DCS6932_225,_DCS6932_226]
WindowsCrack3=[ _DCS6928_137 ,_DCS6928_170,_DCS6928_203,_DCS6928_236,_DCS6928_270,_DCS6928_303,_DCS6928_336,_DCS6928_369]
WindowsCrack4=[ _DCS6928_447 ,_DCS6928_448,_DCS6928_449,_DCS6928_482,_DCS6928_483,_DCS6928_484,_DCS6928_485,_DCS6928_486,_DCS6928_487,_DCS6928_488,_DCS6928_518]

# endregion

crack=[Crack1,Crack2,Crack3,Crack4] # list of cracks to check
windows=[WindowsCrack1,WindowsCrack2,WindowsCrack3,WindowsCrack4] # list of windows for each subimage
resultlis=[]

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
    for y in range(56, image.shape[0], stepSize):
        for x in range(28, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


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


def selectimg(crack,i):
        image = cv2.imread(crack+'.jpg')
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image,img

def joinwindows(img,windows,i):
    resultImg = img.copy() * 0
    clone = img.copy()

    for j in range(0, len(windows[i])):
        x=windows[i][j][0]                      # x coordinate of the upper left corner of the window to evaluate
        y=windows[i][j][1]                      # y coordinate of the upper left corner of the window to evaluate
        window = img[y:y + winH, x:x + winW]  # window to evaluate from x and y coord
        red = np.histogram(window.ravel(), bins=256, range=[0, 256])   # hist for the window
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)    # rectangle to see where we are in the image

        trhs = balanced_hist_thresholding(red)
    # ====================================================================================================================================================================================================
        if i==1 and j==6:
            trhs=55
    # ====================================================================================================================================================================================================
        ret, thresh = cv2.threshold(window, trhs, 255, cv2.THRESH_BINARY)
        thresh=invert(thresh)
        xx = 0
        yy = 0
        for k in range(x, x+window.shape[1]):

            for l in range(y, y+window.shape[0]):
                resultImg[l,k]=thresh[xx,yy]
                xx+=1
            yy += 1
            xx=0

        plt.figure('window hist trsh', figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(img)
        plt.subplot(2, 2, 2)
        plt.imshow(window, cmap='gray')
        plt.subplot(2, 2, 3)
        # plt.hist(window.ravel(), 256, [0, 256])
        plt.imshow(clone)
        plt.subplot(2, 2, 4)
        plt.imshow(resultImg, cmap='gray')



    return resultImg,window

for h in range (0,len(crack)):
    for i in range (0, len(crack[h])):
        pathsubfolder='\Crack '+str(h+1)    # Name of the folder where the predicted subimages detected as cracked are located
        path2 = path + pathsubfolder        # Complete the path name with the folder name
        os.chdir(path2)                     # Access the path

        selectedimage,imageBW=selectimg(crack[h][i],i)  # Get the subimage (selected image) and turns it into greyscale (imageBW)
        resultado,wind=joinwindows(imageBW,windows[h],i)# Get the different windows for each subimage together to create an image with only the crack
        resultlis.append(resultado)

        resultImage=dict.cleanimage(resultado,3)    # Takes the image with only the crack and removes small objects according to the specified size
        resname='res'+crack[h][i]                   # name of the image that will be saved
        dict.imgSaving(path2, resname, resultImage) # The image where small object have been removed is saved in the path

        # widths: list of of widths per pixel of skeleton
        # coordsk: List of coordinates of the skeleton's pixels
        # skframes: Skeleton image
        # edgesframes: Image of the edges of the crack
        # completeList: List of results, creates a vector with x,y coord of the skeleton and the corresponding width for
        #               that skeleton pixel and in mm according to the measure of pixel width
        widths,coordsk,skframes,edgesframes,completeList = dict.CrackWidth(resultImage//255,pixel_width)
        skframesname = 'skframes_' + crack[h][i]  # name of the skeleton image that will be saved
        edgesframesname = 'edgesframes_' + crack[h][i]  # name of the edges image that will be saved
        completeListname = 'completeList_' + crack[h][i]+'.txt'  # name of the image that will be saved
        dict.imgSaving(path2, skframesname, skframes)  # the image where skeleton is saved in the path
        dict.imgSaving(path2, edgesframesname, edgesframes)  # the image where edges of the crack is saved in the path
        with open(path2 + '//'+completeListname, "w") as output:# saves the list as a txt file
            output.write(str(completeList))



