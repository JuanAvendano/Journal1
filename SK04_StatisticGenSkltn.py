"""
Created on Tue may 09 2023
@author: jca

Takes the results of the previous step and compares them with the results of the RefMeasures file to compute the error
and create the final results report that is saved in the results folder


"""

from scipy.spatial import KDTree
import os
import time
import numpy as np


start_time = time.time()

def Statistics(k,save_info,start,end):
    # # ================================================================================================================
    # Inputs
    # # ================================================================================================================
    # Method, used to know where the final subimg is located and where files need to be saved
    method_threshold = k
    # If the file including the reference measurements, method measurements and errors wants to be saved
    save_info=save_info

    # Path for the different possible cracks
    path = r'C:\Users\juanc\OneDrive - KTH\Journals\01-Quantification\Image_list'
    # Path for the file to be saved
    path2= path+'\\Error Results'

    # # ================================================================================================================
    # 2. Process
    # # ================================================================================================================
    # 2.1 Get Reference measurements
    # # ===============================================================================================================
    # List where all results will be stored, first 5 columns correspond to the reference points and measurements, the
    # following 5 for the method measurements and the last for the error.
    full_list=[]

    # Access Path
    os.chdir(path)

    # opening the file in read mode
    with open('RefMeasures.txt') as f:
        # Read in the header line
        header = f.readline().strip()
        # List of points, each point represented as a tuple of (crack,measurement, x, y, width_mm)
        full_list = [tuple(map(float, line.strip().split())) for line in f]

    # 2.2 Create list with reference measurements, measurements made using MAD and corresponding error
    # # ===============================================================================================================
    # Initialize a counter
    ind=0

    # Iterate through each value of n corresponding to the number of cracks
    for n in range(start, end+1):
        # 2.2.1. Paths arrangement
        # # =============================================================================================================...
        # Name of the folder of the crack
        pathsubfolder = '\Crack ' + str(n)

        # Complete the path name with the folder name
        path3 = path + pathsubfolder

        # Path for the specified MAD value used
        pathMAD = path3 + '\MAD k=' + str(method_threshold)

        # 2.2.2. Get the MAD results for each crack
        # # =============================================================================================================...
        # Find the file called Points_for_results.txt in the current folder
        points_filename = os.path.join(pathMAD, '000_Points_for_results.txt')

        # List of points, each point represented as a tuple of (x, y, width_px, width_mm, danger_group)
        points_list = []

        # Read in the contents of Points_for_results.txt
        with open(points_filename) as f:
            # Ignore the header line
            f.readline()
            for line in f:
                try:
                    x, y, width_px, width_mm, danger_group = map(float, line.strip().split())
                except ValueError:
                    x, y, width_px, width_mm, danger_group = np.nan, np.nan, np.nan, np.nan, np.nan
                points_list.append((x, y, width_px, width_mm, danger_group))

        # 2.2.3. Modify the full_list with the new points
        # # =========================================================================================================...
        for i, point in enumerate(points_list):
            full_list[ind] = list(full_list[ind])  # convert tuple to list
            full_list[ind][5:10] = point  # update columns 6-10 with the new values
            full_list[ind] = tuple(full_list[ind])  # convert list back to tuple
            ind+=1

    # 2.3. Determine the error between reference measurements and method measurements
    # # =============================================================================================================...
    for i in range(len(full_list)):
        try:
            ref_measure=full_list[i][4]
            MAD_measure=full_list[i][8]
            error = round((MAD_measure-ref_measure)/ref_measure*100, 2) # round the error value to 2 decimal places
        except ValueError:
            error = np.nan

        full_list[i] = list(full_list[i])  # convert tuple to list
        full_list[i].append(error)   # add column 11 with the error values
        full_list[i] = tuple(full_list[i])  # convert list back to tuple

    # # ================================================================================================================
    # 3. Saving results
    # # ================================================================================================================

    if save_info == True:
        # Writes the updated coordinates to a new text file
        results_filename = '000_Full_List_Results MAD k=' + str(method_threshold)+'.txt'
        # Columns names
        column_names = ['Crack','Measurement','X position','Y position','Manual measurement (mm)','Y position',
                        'X position','Width(pxl)','Width(mm)','DG','Error']
        header = '\t'.join(column_names)
        with open(path2 + '//' + results_filename, "w") as output:
            # Write the header line to the output file
            output.write(header + '\n')
            # Write the updated coordinates to the output file
            for result in full_list:
                result='\t'.join([str(x) for x in result])
                output.write(result + "\n")


# # ====================================================================================================================
# Inputs
# # ====================================================================================================================
# Must be 0 if method is Balanced histogram. If it is MAD the value is the threshold value
k = 2
# Cracks to be studied
start = 1
end = 29
# If the merged list of x,y coordinates and widths want to be saved as text file
save_info=True
# Batch process
Statistics(k,save_info,start,end)

# Finish time counter
end_time = time.time()
# Total time for the code
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
