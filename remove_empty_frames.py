import os

def find_empty_folders(base_folder):
    """
    Walks through the base folder and identifies subfolders that are empty.
    
    Args:
        base_folder (str): Path to the base folder to search for empty subfolders.
        
    Returns:
        List[str]: A list of paths to empty subfolders.
    """
    empty_folders = []
    
    # Walk through the base folder
    for root, dirs, files in os.walk(base_folder):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # Check if the folder is empty (i.e., no files inside it)
            if not os.listdir(dir_path):  # os.listdir returns empty list if the folder is empty
                empty_folders.append(dir_path)
    
    return empty_folders

def remove_empty_folders(empty_folders):
    """
    Removes the provided empty folders from the filesystem.
    
    Args:
        empty_folders (List[str]): A list of paths to empty folders that need to be removed.
    """
    for folder in empty_folders:
        try:
            os.rmdir(folder)  # os.rmdir removes the folder only if it's empty
            print(f"Removed empty folder: {folder}")
        except OSError as e:
            print(f"Error removing folder {folder}: {e}")

# Example usage:
for i in range(1, 10):
    base_folder = f'/scratch-shared/ivanmiert/overview/frames_folder_base/chunk_{i}'  # Update this path

    # Step 1: Find empty folders
    empty_folders = find_empty_folders(base_folder)
    if empty_folders:
        print("Empty folders found:")
        for folder in empty_folders:
            print(folder)
    else:
        print("No empty folders found.")



    # Optional: Step 2: Remove empty folders (uncomment if you want to remove them)
    remove_empty_folders(empty_folders)