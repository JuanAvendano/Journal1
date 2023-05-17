"""
Created on Tue may 09 2023
@author: jca
Generate results per crack:codigo para unir los diferentes txt files de cada sub imagen donde esta la informacion de las
coordenadas del esqueleto y los width.
"""

import time
from UnitMergeTEXTfiles import mergetxtfiles


start_time = time.time()

# # ====================================================================================================================
# Inputs
# # ====================================================================================================================
# Must be 0 if method is Balanced histogram. If it is MAD the value is the threshold value
k = 3

# If the merged list of x,y coordinates and widths want to be saved as text file
save_info = True

# Batch process
for i in range(1,14):
    mergetxtfiles(i, k, save_info)

# Finish time counter
end_time = time.time()
# Total time for the code
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")