"""
Created on Tue may 09 2023
@author: jca

Generate the full process for a certain value of k using MAD, going from:
-Generating results per crack:skeleton subimages, edges subimages, crack binary subimgs, coord and widths per subimage,
 final subimages (with crack, edges and skeleton pixels), and final images reconstructed with and without divisions
-Merge the text files with the skeleton coord and the associated width per pixel
-Put the reference points in the final image as green pixels
-Determine the distance from those reference points to the closest points of each skeleton
-Performs the error calculation for all the method measurements against the reference measurements

"""

import time
from UnitResults import unit_result
from UnitMergeTEXTfiles import mergetxtfiles
from UnitRefPoints import RefPoints
from UnitDist_RefPoints_Skeleton import dist_ref_points_skeleton
from UnitStatistic import Statistics
from UnitMADFull_subimg import MADfull_subimg

start_time = time.time()
# # ====================================================================================================================
# Inputs
# # ====================================================================================================================
# Must be 0 if method is Balanced histogram. If it is MAD the value is the threshold value
k =2
# pixel_width in mm
pixelwidth = 0.08
# Range of cracks that are gonna be studied
start=5
end=6 #(Must be 1 more than the desired number)

# Analisis performed using the windows in each sub image or using the full sub image as a whole
subimgwindows=True
fullsubimg=False

# If the info related to x,y coordinates and widths want to be saved as text file
save_info_Results = True
# If image without small object, skeletons, edges want to be saved as png
saveimgparts = True
# If the generated final subimages want to be saved
savesubimg = True
# If the generated images want to be saved
saveimg = True

#For full sub img:
save_finalsubimg=True
save_final_full_subimage=True

# If the merged list of x,y coordinates and widths want to be saved as text file
save_info_MergeText = True

# Orientation of the crack
horizontal = True
# If the generated final marked image want to be saved
save_img_RefPoints = True
# If the generated reference point list want to be saved
saveRef_Points_list = True

# If the info related to x,y coordinates and widths want to be saved as text file for the skeleton points close to the
#reference points
save_info_Dist_RP_Skl = True

# If the file including the reference measurements, method measurements and errors wants to be saved
save_info_Stats = True


# # ====================================================================================================================
# 1. Results per crack
# # ====================================================================================================================

# Batch process on sub image windows
for x in range(start,end):
    unit_result(x, k, pixelwidth, save_info_Results, saveimgparts, savesubimg, saveimg)


# # ====================================================================================================================
# 2. Merge Text files
# # ====================================================================================================================
# Batch process
for i in range(start,end):
    mergetxtfiles(i, k, save_info_MergeText)

# # ====================================================================================================================
# 3. Create Reference Points in the obtained Crack#MAD image (final img) as green pixels
# # ====================================================================================================================
# Batch process
for i in range(start,end):
    horizontal = True
    if i in [1, 3, 7, 11]:
        horizontal = False
    RefPoints(i, k, horizontal, save_img_RefPoints, saveRef_Points_list)


# # ====================================================================================================================
# 4. Calculates the distances from the RefPoints to the closest Skeleton point and generates a list with those points
# # ====================================================================================================================
# Batch process
for i in range(start,end):
    dist_ref_points_skeleton(i, k, save_info_Dist_RP_Skl)

# # ====================================================================================================================
# 5. Calculates the errors for the diferent measurements
# # ====================================================================================================================
# List of results using the batch results obtained previously
Statistics(k,save_info_Stats)


# Finish time counter
end_time = time.time()
# Total time for the code
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")