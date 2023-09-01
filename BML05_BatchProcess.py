"""
Created on Thr aug 24 2023
@author: jca

Script to run multiple scripts together

"""

import subprocess
import time
from BML01_UnitResults import unit_result_BML
from BML02_PutRefPoints import PutRefPointsBML
from BML03_distRefPointstoSklt import dist_ref_points_BML
from BML04_StatisticBML import Statistics

start_time = time.time()
# # ====================================================================================================================
# Inputs
# # ====================================================================================================================
# Must be 0 if method is Balanced histogram. If it is MAD the value is the threshold value
ks=[2.5,2.2,2]
for i in range(len(ks)):
    k = ks[i]
    # Cracks to be studied
    start=1
    end=29
    # pixel_width in mm
    pixelwidth = 0.08


    #BML01
    for x in range(start, end+1):
        unit_result_BML(x, k, pixelwidth, save_subimg=False, save_info=True, saveimgparts=False,
                                  save_preliminaryimg=True, saveimg=True)

    # BML02
    for i in range(start, end + 1):
        horizontal = True
        if i in [1, 3, 7, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]:
            horizontal = False
        PutRefPointsBML(i, k, horizontal, saveimg=True, saveref_points_list=False)

    # BML03
    for i in range(start, end + 1):
        dist_ref_points_BML(i, k, save_info=True)

    # BML04

    Statistics(k=k, save_info=True,start=start,end=end)

# Finish time counter
end_time = time.time()
# Total time for the code
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")