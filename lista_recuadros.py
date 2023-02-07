# """
# Este codigo esta hecho para pasar por las fotos de las grietas y determinar cuales recuadros de 28x28 contienen grietas.
# Al encontrar resultados positivos hay que poner en el Console: lista_cuadros.append([x,y]) para agregar las coordenadas
# como elementos de una lista. Los elementos de la lista son el resultado y se estan copiando y pegando en un archivo de
# excel para tener la lista completa. El archivo se llama "lista recuadros" en C:\Users\juanc\OneDrive - KTH\Journals\01-Quantification
# """



import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import face
from PIL import Image, ImageDraw
import os
import cv2
import xlsxwriter
from old_Crack_width_calculation import CrackWidth
from numpy import empty
import seaborn as sns
from scipy.stats import norm

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
        for x in range(0, image.shape[1], stepSize):#28,56,84,112,140,168,196,224,252,280,308,336,364,392,420,448,476,504,532,560,588,616,644,672,700
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

path = r'C:\Users\juanc\OneDrive - KTH\Journals\Data\to review\Cracked'
os.chdir(path)  # Access the path
name='_DCS7230_582.jpg'
image = cv2.imread(name)
pixel_width=0.1
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
winW = 28
winH = 28
lista_cuadros=[]



for (x, y, window) in sliding_window(img, stepSize=28, windowSize=(winW, winH)):
    # Histogram and balanced threshold for window in the initial sub-image

    red = np.histogram(window.ravel(), bins=256, range=[0, 256])
    clone = img.copy()
    # Creates rectangle in the coordinates over the clone to see it in the ploted image
    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    #Plot the image with the window, the pixels in the window, histogram of the window
    plt.figure('window hist trsh', figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(clone, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(window, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.hist(window.ravel(), 256, [0, 256])
    plt.subplot(2, 2, 4)
    sns.distplot(window.ravel(), fit=norm, kde=False)
print(name,';',lista_cuadros)

# lista_cuadros.append([x,y]) # poner este comando en el Console al ir viendo resultados positivos

# lista_cuadros[0][0]=name
# lista_cuadros[0][1]=''
# lista_cuadros[1][0]='x'
# lista_cuadros[1][1]='y'