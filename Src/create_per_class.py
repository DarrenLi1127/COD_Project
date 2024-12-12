import os
import shutil

# Specify the source directory containing your images
source_dir = "/school/CSCI_2470/COD_Project/results/predictions_archive_80_per_class"

# Get all files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith('.png'):
        # Extract the class name (assuming it's the second to last part when split by '-')
        class_name = filename.split('-')[-2]
        
        # Create the subfolder path
        subfolder_path = os.path.join(source_dir, class_name)
        
        # Create the subfolder if it doesn't exist
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
        
        # Move the file to its corresponding subfolder
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(subfolder_path, filename)
        shutil.move(source_file, destination_file)