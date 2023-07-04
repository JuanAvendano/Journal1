"""
Created on
@author: jca
Este codigo esta hecho para pasar por las fotos de las grietas y determinar cuales recuadros de 28x28 contienen grietas.
Se genera una imagen con las windows en la subimagen y su correspondientes numeros de window.
parar en lista_cuadros=... y en el Console poner nn=[] y adentro los numeros de las windows deseadas.
Las coordenadas de los elementos de la lista son el resultado y se deben copiar y pegar en un archivo de
excel para tener la lista completa. El archivo se llama "lista recuadros" en la carpeta Image List

"""

import numpy as np
import cv2
import seaborn as sns
from scipy.stats import norm
from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib.pyplot as plt

path = r'C:\Users\juanc\OneDrive - KTH\Journals\01-Quantification\Image_list'

# Number of the cracked that is going to be processed
n =11
# Name of the folder where the information of the desired crack is located
pathsubfolder = '\Crack ' + str(n)+'\\00_Cracked_subimg\\'
path2 = path + pathsubfolder  # Complete the path name with the folder name
# Access the path
os.chdir(path2)
pixel_width=0.1
winW = 28
winH = 28
lista_cuadros=[]
lista_completa=[]

name= ['_DCS7221_577']
for k in range(len(name)):
    image = Image.open(name[k]+'.png')
    img = image.convert('L')
    list=[[x, y] for x in range(0, 224, 28) for y in range(0, 224, 28)]
    clone = img.copy()

    for i,(x, y) in enumerate(list):
        # x_offset = col * max(widths)
        # y_offset = row * max(heights)
        # result.paste(image, (x_offset, y_offset))
        # Draw bounding box around image
        draw = ImageDraw.Draw(clone)
        draw.rectangle((x, y, x + winW, y + winH), outline='white')

        # Add label on top of image
        label = f"{i,}"
        font = ImageFont.truetype("arial.ttf", 8)
        text_width, text_height = draw.textsize(label, font)
        text_x = x + (winW - text_width) // 2
        text_y = y + text_height
        # draw.rectangle((text_x - 5, text_y - 5, text_x + text_width + 5, text_y + text_height + 5), fill=(0, 0, 0, 128))
        draw.text((text_x, text_y), label, font=font, fill='white')


    #Plot the image with the window, the pixels in the window, histogram of the window
    plt.figure('window ', figsize=(10,10))
    plt.imshow(clone)

    nn=[]
    lista_cuadros=[list[numero] for numero in nn ]
    lista_completa.append([name[k], lista_cuadros,'x'])
    print(name[k],';',lista_cuadros)

print(lista_completa)
# parar en lista_cuadros=... y en el Console poner nn=[] y adentro los numeros de las windows deseadas

#
# for (x, y, window) in lista_cuadros:
#     # Histogram and balanced threshold for window in the initial sub-image
#
#     red = np.histogram(window.ravel(), bins=256, range=[0, 256])
#     clone = img.copy()
#     # Creates rectangle in the coordinates over the clone to see it in the ploted image
#     cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
#     #Plot the image with the window, the pixels in the window, histogram of the window
#     plt.figure('window hist trsh', figsize=(10, 10))
#     plt.subplot(2, 2, 1)
#     plt.imshow(clone, cmap='gray')
#     plt.subplot(2, 2, 2)
#     plt.imshow(window, cmap='gray')
#     plt.subplot(2, 2, 3)
#     plt.hist(window.ravel(), 256, [0, 256])
#     plt.subplot(2, 2, 4)
#     sns.distplot(window.ravel(), fit=norm, kde=False)
# print(name,';',lista_cuadros)
