import os
import shutil
import pandas as pd

def distribute_images_by_class(root_folder, destination_folder, excel_file):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(excel_file, header=None)
    
    # Create a dictionary to map subfolders to their respective classes
    class_mapping = {}
    
    # Iterate through each column to get the class name and subfolders
    for column in df.columns:
        class_name = str(df.iloc[0, column])  # First row is the class name
        
        # Extract subfolder names, dropping any NaN values, and convert to string
        subfolders = df.iloc[1:, column].dropna().apply(str).tolist()
        
        for subfolder in subfolders:
            class_mapping[subfolder] = class_name
    print(class_mapping)

# Create the destination folders for each class if they don't exist
    for class_name in set(class_mapping.values()):
        class_folder = os.path.join(destination_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)
    
    # Traverse the subfolders in the root directory
    for subdir, _, files in os.walk(root_folder):
        subfolder_name = os.path.basename(subdir)
        
        # Check if this subfolder should be moved to a class
        if subfolder_name in class_mapping:
            class_name = class_mapping[subfolder_name]
            class_folder = os.path.join(destination_folder, class_name)
            
            for file_name in files:
                # Construct the full file path
                file_path = os.path.join(subdir, file_name)
                
                # Check if the file is an image
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    # Create a new unique file name
                    new_file_name = f"{subfolder_name}_{file_name}"
                    
                    # Create the full path for the new file location
                    new_file_path = os.path.join(class_folder, new_file_name)
                    
                    # Copy the image to the class folder with the new name
                    shutil.copy2(file_path, new_file_path)
                    print(f"Copied {file_path} to {new_file_path}")

# Usage
root_folder = "D:\\Footage\\Classifier\\train_color"
destination_folder = 'D:\\Footage\\Classifier\\train_color_2'
excel_file = 'D:\\GITHUB\\AdvancedDataPreProcessingMethods\\NEW_OBJECT\\kleuren_2.xlsx'
distribute_images_by_class(root_folder, destination_folder, excel_file)