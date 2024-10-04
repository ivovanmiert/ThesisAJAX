import os
import glob

# Define the folder path
folder_path = "/scratch-shared/ivanmiert/overview/ball_detection_processed"

# Use glob to find all files ending with '_processed.csv'
files_to_delete = glob.glob(os.path.join(folder_path, '*_processed.csv'))

# Loop through and delete each file
for file_path in files_to_delete:
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")