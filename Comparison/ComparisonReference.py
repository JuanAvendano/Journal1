import cv2
import numpy as np
import os
import Dictionary as dicc
import matplotlib.pyplot as plt
from numpy import empty
from skimage.util import invert

def comparison(ref_img_to_select,target_img_to_select,pathREF,pathTarget):


    # Load the reference image and the other image
    ref_img_to_select = ref_img_to_select
    os.chdir(pathREF)
    reference_image = cv2.imread(ref_img_to_select + '.png')
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    target_img_to_select=target_img_to_select
    os.chdir(pathTarget)
    target_img = cv2.imread(target_img_to_select + '.png')
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

    # Create a mask for black pixels in the reference image and yellow pixels in the other image
    black_mask = invert(reference_image) > 0
    yellow_mask = cv2.inRange(target_img, (255, 255, 0), (255, 255, 0))
    red_mask = cv2.inRange(target_img, (255, 0, 0), (255, 0, 0))
    crack_mask = yellow_mask + red_mask
    crack_pixels = crack_mask > 0

    # Sets the image width and heigh
    width = reference_image.shape[0]
    height = reference_image.shape[1]

    yellow_count = 0
    green_count = 0
    orange_count = 0
    white_count = 0

    # Creates the finalsubimage element that will have the crack as white and the rest of the pixels in black
    final_image = empty([width, height, 3], dtype=np.uint8)
    for x in range(0, width):
        for y in range(0, height):
            if black_mask[x, y] and crack_pixels[x, y]:
                final_image[x, y] = [255, 255, 0]  # If pixel is part of Reference and Cracked image, paint yellow (TP)
                yellow_count += 1
            elif black_mask[x, y]:
                final_image[x, y] = [0, 255, 0]  # If pixel is just part of Reference, paint green (FN)
                green_count += 1
            elif crack_pixels[x, y]:
                final_image[x, y] = [255, 128, 0]  # If pixel is just part of Cracked image, paint orange (FP)
                orange_count += 1
            else:
                final_image[x, y] = [0, 0, 0]  # If is background for both, paint white (TN)
                white_count += 1

    uncracked_subimgs=[[1,0],[2,4],[3,8],[4,16],[5,18],[6,18],[7,34],[8,8],[9,13],[10,20],[11,35],[12,12],[13,6],[14,66],
                       [15,50],[16,52],[17,17],[18,37],[19,20],[20,11],[21,42],[22,41],[23,39],[24,64],[25,76],[26,29],
                       [27,14],[28,29],[29,39]]

    white_count = white_count - 224 * uncracked_subimgs[i][1]


    return (final_image, yellow_count,green_count,orange_count,white_count)

save_FINAL_img=True
pathREF = r'C:\Users\jcac\OneDrive - KTH\Journals\01-Quantification\Image_list\Reference\Final references masks'
pathResult= r'C:\Users\jcac\OneDrive - KTH\Journals\01-Quantification\Image_list\Method comparison'

process_BME=False
process_OTSU=True


#MBE part
if process_BME==True:
    Results_summary_MBE = "Summary_MBE.txt"
    Results_summary_path=os.path.join(pathResult, Results_summary_MBE)
    with open(Results_summary_path, "a") as file:
        # Write the column headers
        file.write("Method\tk\tCrack\tTP\tFN\tFP\tTN\n")

        listk=[2,2.2,2.5,3,3.5]
        for x in listk:
            k = x
            pathTarget = r'C:\Users\jcac\OneDrive - KTH\Journals\01-Quantification\Image_list\BME\k='+str(k)
            pathResults_subfolder=pathResult+'\MBE k='+str(k)
            # List of images paths
            refimglist0 = os.listdir(pathREF)
            refimglist = [element.split('.')[0] for element in refimglist0 if "Final_REF" in element and  element.endswith('.png')]
            refimglist=sorted(refimglist, key=lambda im: int((im.split(' ')[-1]).split('.')[0]))

            # List of images paths
            target_imglist0 = os.listdir(pathTarget)
            target_imglist = [element.split('.png')[0] for element in target_imglist0 if "BME" in element and  element.endswith('.png')]
            target_imglist=sorted(target_imglist, key=lambda im: int((im.split(' ')[1]).split('_')[0]))

            for i in range(0, len(refimglist)):

                ref_img_to_select = refimglist[i]
                target_img_to_select=target_imglist[i]

                final_image, TP,FN,FP,TN=comparison(ref_img_to_select, target_img_to_select, pathREF, pathTarget)

                # Save the final image
                cracknumb=ref_img_to_select.split(' ')[-1]
                finalFull_imgname = 'Comparison_BME_k='+str(k)+' Crack ' + cracknumb
                if save_FINAL_img:
                    os.chdir(pathResults_subfolder)
                    # The image is saved in the path
                    dicc.imgSaving(pathResults_subfolder, finalFull_imgname, final_image)

                file.write(f"{'MBE'}\t{k}\t{i+1}\t{TP}\t{FN}\t{FP}\t{TN}\n")












#OTSU part
if process_OTSU==True:
    Results_summary_OTSU = "Summary_OTSU.txt"
    Results_summary_path2=os.path.join(pathResult, Results_summary_OTSU)
    with open(Results_summary_path2, "a") as file:
        # Write the column headers
        file.write("Method\tk\tCrack\tTP\tFN\tFP\tTN\n")

        pathTarget = r'C:\Users\jcac\OneDrive - KTH\Journals\01-Quantification\Image_list\OTSU\OTSU results'
        pathResults_subfolder=pathResult+'\OTSU'
        # List of images paths
        refimglist0 = os.listdir(pathREF)
        refimglist = [element.split('.')[0] for element in refimglist0 if "Final_REF" in element and  element.endswith('.png')]
        refimglist=sorted(refimglist, key=lambda im: int((im.split(' ')[-1]).split('.')[0]))

        # List of images paths
        target_imglist0 = os.listdir(pathTarget)
        target_imglist = [element.split('.')[0] for element in target_imglist0 if "OTSU" in element and  element.endswith('.png')]
        target_imglist=sorted(target_imglist, key=lambda im: int((im.split('CRACK')[1]).split('.')[0]))

        for i in range(0, len(refimglist)):
            ref_img_to_select = refimglist[i]
            target_img_to_select = target_imglist[i]

            final_imageOtsu, TP, FN, FP, TN = comparison(ref_img_to_select, target_img_to_select, pathREF, pathTarget)

            # Save the final image
            cracknumb = ref_img_to_select.split(' ')[-1]
            finalFull_imgname = 'Comparison_Otsu' + ' Crack ' + cracknumb
            if save_FINAL_img:
                os.chdir(pathResults_subfolder)
                # The image is saved in the path
                dicc.imgSaving(pathResults_subfolder, finalFull_imgname, final_imageOtsu)

            file.write(f"{'OTSU'}\t{'0'}\t{i+1}\t{TP}\t{FN}\t{FP}\t{TN}\n")