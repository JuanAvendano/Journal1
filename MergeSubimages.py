"""
Created on Fri march 24 2023
@author: jca
code to merge the result images from MAD into one image and one image with divisions for clarity and in case
checking is needed"""


import Dictionary as dict
import os
import glob




num=29
numfin=30
# Crack dimensions in terms of rows and columns of subimages for each crack (first element is the number of the crack)
crackgeometry = [[1,1,7],[2,5,2],[3,2,8],[4,9,3],[5,5,3],[6,5,3],[7,4,6],[8,5,2],[9,5,4],[10,6,3],[11,6,8],[12,6,2],[13,6,2],[14,8,10],[15,7,8],[16,4,18],[17,3,9],[18,4,13],[19,3,10],[20,3,7],[21,5,11],[22,6,9],[23,4,13],[24,5,17],[25,6,16],[26,4,10],[27,6,4],[28,17,3],[29,15,4]]

for i in range (num,numfin):
    path1 = r'C:\Users\juanc\OneDrive - KTH\Journals\01-Quantification\Image_list\Crack '+str(i)+'\\00_Cracked_subimg\\'
    path2 = r'C:\Users\juanc\OneDrive - KTH\Journals\01-Quantification\Image_list\Crack '+str(i)+'\\01_Uncracked_subimg\\'
    dst=r'C:\Users\juanc\OneDrive - KTH\Journals\01-Quantification\Image_list\Crack '+str(i)+'\\'

    os.chdir(path1)  # Access the path
    img_list1 = glob.glob(path1 +'*.png')  # Images from path to be stitched together
    img_list2 = glob.glob(path2 +'*.png')  # Images from path to be stitched together
    img_list=img_list1+img_list2


    sortedlist = sorted(img_list,key=lambda im: int((im.split('_')[-1]).split('.')[0]))

    nr = crackgeometry[i-1][2]
    nc = crackgeometry[i-1][1]
    # merge sub images
    image = dict.merge_images(sortedlist, nr, nc)
    # merge sub images with their corresponding label
    image_div = dict.merge_images_with_labels(sortedlist, nr, nc)


    newname = 'CRACK.png'       # path.split('\\')[7]+'MAD.png'
    newname2 = 'div CRACK.png'  # path.split('\\')[7]+' divMAD.png'
    image.save(dst + newname)
    image_div.save(dst + newname2)

#