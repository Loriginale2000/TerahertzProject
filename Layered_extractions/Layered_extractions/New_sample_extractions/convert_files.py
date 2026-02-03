import glob
from mat_to_csv import convert_mat_to_csv
import glob

mat_files = glob.glob("New_samples_mat/*.mat")
print(mat_files)
for mat_file in mat_files:
    convert_mat_to_csv(mat_file, 'New_samples_csv')

print('Success!')