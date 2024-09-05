import os

# Specify the folder path
folder_path = '/home/sneha/augmen_uncertainity/intelligent_sampling/final_calculated_structures/'

# Read filenames from .pl file
pl_file_path = '/home/sneha/augmen_uncertainity/intelligent_sampling/delete.pl'
with open(pl_file_path, 'r') as pl_file:
    filenames = [line.strip().split('/')[-1] for line in pl_file]
    print(filenames)

# Remove .xyz files from the specified folder
for filename in filenames:
    xyz_file_path = os.path.join(folder_path, filename)
    if os.path.exists(xyz_file_path):
        os.remove(xyz_file_path)
        print(f'Removed: {xyz_file_path}')
    else:
        print(f'File not found: {xyz_file_path}')
