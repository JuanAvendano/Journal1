"""
Created on Tue may 09 2023
@author: jca
Codigo para determinar las mediciones de width en los correspondientes puntos de referencia. Como las mediciones corres-
ponden a los width calculados en los diferentes puntos del esqueleto, esos puntos pueden no corresponder a los puntos
de referencia asi que el codigo busca cual es el pixel del esqueleto que se encuentra cerca del punto de referencia si
es que hay puntos de esqueleto a por lo menos 5 pixeles de distancia del punto de referencia, de lo contrario se considera
muy lejos y no representativo.

"""

import time
from UnitDist_RefPoints_Skeleton import dist_ref_points_skeleton

start_time = time.time()
# # ================================================================================================================
# Inputs
# # ================================================================================================================
# Method, used to know where the final subimg is located and where files need to be saved
k=3
# If the info related to x,y coordinates and widths want to be saved as text file for the skeleton points close to the
#reference points
save_info = True
# Batch process
for i in range(1,14):
    dist_ref_points_skeleton(i, k, save_info)

# Finish time counter
end_time = time.time()
# Total time for the code
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")