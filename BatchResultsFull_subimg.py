"""
Created on Tue jun 07 2023
@author: jca
Generate results per crack: skeleton subimages, edges subimages, crack binary subimages, coord and widths per subimage,
final subimages (with crack, edges and skeleton pixels), and final images reconstructed with and without divisions

"""
import time
from UnitMADFull_subimg import MADfull_subimg


start_time = time.time()
# # ====================================================================================================================
# Inputs
# # ====================================================================================================================
# Must be 0 if method is Balanced histogram. If it is MAD the value is the threshold value
k = 2.5
# pixel_width in mm
pixelwidth = 0.09
# If
save_finalsubimg = True
# If
save_final_full_subimage = True


for x in range(14, 30):
    MADfull_subimg(x, k, pixelwidth, save_finalsubimg, save_final_full_subimage)

# Finish time counter
end_time = time.time()
# Total time for the code
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
