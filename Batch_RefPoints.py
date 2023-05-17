"""
Created on Tue may 09 2023
@author: jca
Generate results per crack:
Codigo establece los puntos de referencia (mediciones) en las imagenes donde se aplico la cuantificacion. Primero lee la
imagen que correspondiente a crackMarked para cada grieta, en donde se encuentran los puntos donde se hicieron mediciones.
Luego crea una lista con las coordenadas de los pixeles, el orden es segun la orientacion de la foto. Luego pone esos
pixeles en las correspondientes finalimg (ej: Crack5 MAD) como pixeles verdes.

"""

import time
from UnitRefPoints import RefPoints

start_time = time.time()

# # ====================================================================================================================
# Inputs
# # ====================================================================================================================
# Method, used to know where the final subimg is located and where files need to be saved
k = 3
# Orientation of the crack
horizontal = True
# If the generated final marked image want to be saved
save_img = True
# If the generated reference point list want to be saved
saveref_points_list = True

# Batch process
for i in range(1, 14):
    horizontal = True
    if i in [1, 3, 7, 11]:
        horizontal = False
    RefPoints(i, k, horizontal, save_img, saveref_points_list)

# Finish time counter
end_time = time.time()
# Total time for the code
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")