import subprocess
import time

start_time = time.time()
scripts = ['SK02_refpoint.py', 'SK03_closeRefPointstoSklt.py', 'SK04_StatisticGenSkltn.py'] #
# Add more scripts as needed

for script in scripts:
    subprocess.run(['python', script])

# Finish time counter
end_time = time.time()
# Total time for the code
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")