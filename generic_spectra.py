import os
import shutil

source_file = "/home/sneha/XAS-3dtm/fe/preconv6/xanes_test/IHIZEE_C44H43FFeP2.txt"
names_folder = "/home/sneha/augmen_uncertainity/intelligent_sampling/final_calculated_structures"
destination_folder = "/home/sneha/augmen_uncertainity/intelligent_sampling/final_calculated_spectra"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Read the content of the source file
with open(source_file, "r") as f:
    spectrum = f.read()

# Get the list of names from the names folder
name_files = os.listdir(names_folder)

# Generate the files with the generic spectrum
for i in range(940692):
    # Get the name for the current file
    name = name_files[i % len(name_files)][:-4]
    print(name)
    # Set the file path
    file_path = os.path.join(destination_folder, f"{name}.txt")
    # Write the spectrum to the file
    with open(file_path, "w") as f:
        f.write(spectrum)
#print(spectrum)

print(".txt Files generated successfully!")
