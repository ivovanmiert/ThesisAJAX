import os
import shutil

"""This file was used to find the intersection of the features available from the top-down and bottom-up models. 
    This intersection was about 99% of the files, but for some reason, a few files were not available for both, so this intersection was used. 
"""

folder1 = '/scratch-shared/ivanmiert/overview/concatenated_top_down'
folder2 = '/scratch-shared/ivanmiert/overview/concatenated_bottom_up'

folder1_new = '/scratch-shared/ivanmiert/overview/concatenated_top_down_intersection'
folder2_new = '/scratch-shared/ivanmiert/overview/concatenated_bottom_up_intersection'

os.makedirs(folder1_new, exist_ok=True)
os.makedirs(folder2_new, exist_ok=True)


files_in_folder1 = set(os.listdir(folder1))
files_in_folder2 = set(os.listdir(folder2))


common_files = files_in_folder1.intersection(files_in_folder2)

for file in common_files:
    file_path1 = os.path.join(folder1, file)
    file_path2 = os.path.join(folder2, file)
    dest_file1 = os.path.join(folder1_new, file)
    dest_file2 = os.path.join(folder2_new, file)


    if not os.path.exists(dest_file1):
        shutil.copy(file_path1, dest_file1)
        print(f"Copied {file} to {folder1_new}")
    
    if not os.path.exists(dest_file2):
        shutil.copy(file_path2, dest_file2)
        print(f"Copied {file} to {folder2_new}")
