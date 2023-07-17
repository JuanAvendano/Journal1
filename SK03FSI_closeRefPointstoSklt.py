"""
Created on Mon Jul 10 2023
@author: jca

Checks if the skeleton pixels are close to the ref points and generates a points for results text file

"""


from scipy.spatial import KDTree
import os
import time


start_time = time.time()

def dist_ref_points_general_skeleton(n,k,save_info):
    # # ================================================================================================================
    # Inputs
    # # ================================================================================================================
    # Path for the different possible cracks
    path = r'C:\Users\juanc\OneDrive - KTH\Journals\01-Quantification\Image_list'
    # Crack to check
    n=n
    # Method, used to know where the final subimg is located and where files need to be saved
    method_threshold = k
    # If the info related to x,y coordinates and widths want to be saved as text file
    save_info = save_info

    # # ===============================================================================================================.
    # 1. Paths arrangement
    # # ===============================================================================================================.
    # Name of the folder where the predicted subimages detected as cracked are located
    pathsubfolder = '\Crack ' + str(n)
    # Complete the path name with the folder name
    path2 = path + pathsubfolder
    # Path where results will be saved if using MAD
    pathMAD = path2 + '\MAD k=' + str(method_threshold)+' full_subimg'

    path3 = pathMAD
    # Access the path
    os.chdir(path3)

    # # ================================================================================================================
    # 2. Process
    # # ================================================================================================================
    # 2.1 Complete list merged
    # # ===============================================================================================================
    # opening the file in read mode
    with open('000_completeList_Crack' + str(n) +' full_subimg.txt') as f:
        # Read in the header line
        header = f.readline().strip()

        # List of points, each point represented as a tuple of (x, y, width_px, width_mm, danger_group)
        CompleteList = [tuple(map(float, line.strip().split())) for line in f]

    # Create a list with only the coordinates
    points = [(x_coord, y_coord) for x_coord, y_coord, _, _, _ in CompleteList]

    # Create a separate list of tuples for the additional information
    additional_info = [(width_px, width_mm, danger_group) for _, _, width_px, width_mm, danger_group in CompleteList]

    # 2.2 Reference points
    # # ===============================================================================================================
    os.chdir(path2)
    # opening the file in read mode
    with open('ReferencePoints_Crack'+str(n)+'.txt') as f:
        # skip the first line
        next(f)
        # List of reference points, each point represented as a tuple of (x, y)
        ref_points = [tuple(map(float, line.strip().split())) for line in f]

    # Construct a k-d tree from the list of points
    tree = KDTree(points)

    # list of tuples to store results
    results = []

    # Find the closest point to each reference point
    for ref_point in ref_points:
        dist, ind = tree.query(ref_point)
        if dist < 5:
            closest_point = points[ind]
            additional_info_for_closest_point = additional_info[ind]
            # Gets the info of the line into variables
            xcoord, ycoord, w, wmm, d = CompleteList[ind]
            # Appends the variables in the result list
            results.append(f"{xcoord}\t{ycoord}\t{w}\t{wmm}\t{d}")
            print(f"{ref_point} is close to {closest_point[:2]} (distance = {dist}), additional info: {additional_info_for_closest_point}")
        else:
            results.append(f"{None}\t{None}\t{None}\t{None}\t{None}")
            print(f"{ref_point} is not in the list of points and there are no points close by")

    # # ================================================================================================================
    # 3. Saving results
    # # ================================================================================================================

    if save_info == True:
        # Writes the updated coordinates to a new text file
        results_filename = "000_Points_for_results.txt"
        with open(path3 + '//' + results_filename, "w") as output:
            # Write the header line to the output file
            output.write(header + '\n')
            # Write the updated coordinates to the output file
            for result in results:
                output.write(result + "\n")


# # ====================================================================================================================
# Inputs
# # ====================================================================================================================
listk = [2.2, 3, 3.5]
for x in listk:
    # Must be 0 if method is Balanced histogram. If it is MAD the value is the threshold value
    k = x
    # Cracks to be studied
    start = 1
    end = 29
    # If the merged list of x,y coordinates and widths want to be saved as text file
    save_info=True
    # Batch process
    for i in range(start, end + 1):
        dist_ref_points_general_skeleton(i, k, save_info)


# Finish time counter
end_time = time.time()
# Total time for the code
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")