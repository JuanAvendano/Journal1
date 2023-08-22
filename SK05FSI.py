"""
Created on Tue may 09 2023
@author: jca

Script to run multiple scripts together

"""

import subprocess
import time

start_time = time.time()
scripts = ['SK02FSI_refpoint.py', 'SK03FSI_closeRefPointstoSklt.py', 'SK04FSI_StatisticGenSkltn.py'] #
# Add more scripts as needed

for script in scripts:
    subprocess.run(['python', script])

# Finish time counter
end_time = time.time()
# Total time for the code
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")