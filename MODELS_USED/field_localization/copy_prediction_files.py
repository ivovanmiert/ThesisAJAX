import os
import shutil

def copy_specific_files_with_subfolder_name(root_folder, file_extension, destination_folder):
    """
    Copy specific files (with a given file extension) from each subfolder in the root folder
    to a designated destination folder, appending subfolder name to each copied file.

    Args:
    - root_folder (str): Path to the root folder on the remote machine (HPC).
    - file_extension (str): Extension of the files to search for (e.g., '.txt').
    - destination_folder (str): Path to the destination folder on your local machine.

    Returns:
    - None
    """
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Traverse through the root folder and its subfolders
    for dirpath, _, filenames in os.walk(root_folder):
        # Filter files by extension and copy them to the destination folder
        for filename in filenames:
            if filename.endswith(file_extension):
                source_file_path = os.path.join(dirpath, filename)
                
                # Create a unique filename by appending subfolder name
                subfolder_name = os.path.basename(dirpath)  # Get the name of the subfolder
                destination_filename = f"{subfolder_name}_{filename}"  # Append subfolder name to filename
                destination_file_path = os.path.join(destination_folder, destination_filename)

                # Copy the file to the destination folder
                shutil.copyfile(source_file_path, destination_file_path)
                print(f"Copied '{source_file_path}' to '{destination_file_path}'")

# Example usage:
if __name__ == "__main__":
    root_folder = '/scratch-shared/ivanmiert/calibration'  # Replace with the path to your HPC root folder
    file_extension = '.pth'  # Specify the file extension of the files you want to copy
    destination_folder = '/home/ivanmiert/sportlight_folder/soccernet-calibration-sportlight/src/models/hrnet/predictions_saved'  # Replace with the path to your local destination folder

    copy_specific_files_with_subfolder_name(root_folder, file_extension, destination_folder)