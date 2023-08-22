"""
Created on Tue may 09 2023
@author: jca
Generate results per crack: skeleton subimages, edges subimages, crack binary subimages, coord and widths per subimage,
final subimages (with crack, edges and skeleton pixels), and final images reconstructed with and without divisions

"""
import time
from UnitResults import unit_result


start_time = time.time()
# # ====================================================================================================================
# Inputs
# # ====================================================================================================================
# Must be 0 if method is Balanced histogram. If it is MAD the value is the threshold value
k = 3
# Cracks to be studied
start=14
end=29
# pixel_width in mm
pixelwidth = 0.08
# If the info related to x,y coordinates and widths want to be saved as text file
save_info = True
# If image without small object, skeletons, edges want to be saved as png
saveimgparts = True
# If the generated final subimages want to be saved
savesubimg = True
# If the generated images want to be saved
saveimg = True

for x in range(start, end+1):
    unit_result(x, k, pixelwidth, save_info, saveimgparts, savesubimg, saveimg)


# Finish time counter
end_time = time.time()
# Total time for the code
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")


