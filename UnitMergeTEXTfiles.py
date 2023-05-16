"""
Created on Mon april 24 2023
@author: jca
codigo para unir los diferentes txt files de cada sub imagen donde esta la informacion de las coordenadas del
esqueleto y los width.
"""
import os
import glob
import time


start_time = time.time()
def mergetxtfiles(n,k,save_info):
    # # ====================================================================================================================
    # Inputs
    # # ====================================================================================================================
    # Path for the different possible cracks
    path = r'C:\Users\juanc\OneDrive - KTH\Journals\01-Quantification\Image_list'
    # Number of the cracked that is going to be processed
    n = n
    # Method, used to know where the resulting files need to be saved
    method_threshold = k
    # Dimensions of sub images (must be squared)
    width = height = 224
    # If the merged list of x,y coordinates and widths want to be saved as text file
    save_info = save_info

    # region crack info
    # Crack dimensions in terms of rows and columns of subimages for each crack (first element is the number of the crack,
    # second number is the number of columns and third is rows)
    crackgeometry = [[1, 1, 7], [2, 5, 2], [3, 2, 8], [4, 9, 3], [5, 5, 3], [6, 5, 3], [7, 4, 6], [8, 5, 2], [9, 5, 4],
                     [10, 6, 3], [11, 6, 8], [12, 6, 2], [13, 6, 2],[14,6,10]]
    # endregion

    # # ================================================================================================================
    # 1. Paths arrangement
    # # ================================================================================================================
    # Name of the folder where the predicted subimages detected as cracked are located
    pathsubfolder = '\Crack ' + str(n)
    # Complete the path name with the folder name
    path2 = path + pathsubfolder
    # Path where results will be saved if using MAD
    pathMAD = path2 + '\MAD k=' + str(method_threshold)

    path3 = pathMAD
    # Path where the uncracked sub images are for the current crack
    path4 = path2 + '\\01_Uncracked_subimg\\'

    # # ================================================================================================================
    # 2. Process
    # # ================================================================================================================
    # 2.1 Sorted list of subimages for the main image
    # # ===============================================================================================================
    # List of cracked processed subimages (finalsubimg) paths
    subimglist = [os.path.join(path3, f) for f in os.listdir(path3) if "finalsubimg" in f and ".png" in f]
    # List of uncracked subimages paths
    uncrksubimglist = glob.glob(path4 + '*.png')
    # Addition of uncracked subimages path lists
    subimglist = subimglist + uncrksubimglist
    # List sorted in ascending order
    sortedlist = sorted(subimglist, key=lambda im: int((im.split('_')[-1]).split('.')[0]))

    # 2.2 X and Y offset for each sub image
    # # ===============================================================================================================
    # Number of columns in the image
    nc = crackgeometry[n-1][1]
    # Number of rows in the image
    nr = crackgeometry[n-1][2]

    # Create a list to store the x and y offsets for each sub-image
    list_subimg_offset = [[0, 0, 0] for _ in range(len(sortedlist))]
    for x in range(len(sortedlist)):
        list_subimg_offset[x][0] = sortedlist[x].split('_DCS')[1].split('.')[0]

    # Determine x_offset and y_offset for each subimg
    for row in range(nr):
        for col in range(nc):
            index = row * nc + col
            if index >= len(sortedlist):
                break
            x_offset = col * width
            y_offset = row * height
            list_subimg_offset[index][1] = x_offset
            list_subimg_offset[index][2] = y_offset


    # 2.3 Updated coordinate
    # # ===============================================================================================================

    # Create a list to store the updated coordinates for each sub-image
    new_coords = []
    # Create a list of text files in the folder
    infolist = [os.path.join(path3, f) for f in os.listdir(path3) if "completeList" in f]
    # Create a list with the text's subimage numbers
    infolistnumbers = [f.split('__DCS')[1].split('.')[0] for f in infolist]

    # Looks for the x and y offsets for each subimage, then reads each txt file and add the offsets to the x and y coord.
    for index in range(len(infolist)):
        subimg_filename = infolist[index]
        x_offset = 0
        y_offset = 0
        # Looks for the offsets by comparing the list of txt files (in just numbers) and the list with the offsets.
        for x in range(len(list_subimg_offset)):
            if list_subimg_offset[x][0] == infolistnumbers[index]:
                x_offset = list_subimg_offset[x][1]
                y_offset = list_subimg_offset[x][2]
                break
        # Opens the corresponding file, adds the offsets to x and y and stores the information in the new_coords list
        with open(subimg_filename, "r") as f:
            # Read in the header line
            header = f.readline().strip()
            # Read in the coordinates for each pixel
            for line in f:
                ycoord, xcoord, w, wmm, d = line.strip().split("\t")
                xcoord = int(float(xcoord)) + x_offset
                ycoord = int(float(ycoord)) + y_offset
                new_coords.append(f"{xcoord}\t{ycoord}\t{w}\t{wmm}\t{d}")


    # # ================================================================================================================
    # 3. Saving results
    # # ================================================================================================================

    if save_info == True:
        # Writes the updated coordinates to a new text file
        merged_filename = "CompleteMergedList.txt"
        with open(path3 + '//' + merged_filename, "w") as output:
            # Write the header line to the output file
            output.write(header + '\n')
            # Write the updated coordinates to the output file
            for coords in new_coords:
                output.write(coords + "\n")


# Finish time counter
end_time = time.time()
# Total time for the code
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")