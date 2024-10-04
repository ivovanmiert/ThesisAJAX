import os
import glob
import argparse
import subprocess
import json
import csv




def load_subfolders_from_json(json_file_path):
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            # Extract the subfolder values from each item in the list
            subfolders = [entry['subfolder'] for entry in data if 'subfolder' in entry]
            return subfolders
    except Exception as e:
        print(f"Error loading or parsing JSON file: {e}")
        return []
    
def run_openpifpaf_on_folder(input_folder, temp_json_dir, batch_size=1):
    long_edge = 1280
    # Run OpenPifPaf prediction
    command = [
        "python3", "-m", "openpifpaf.predict",
        "--glob", f"{input_folder}/*.jpg",
        "--batch-size", str(batch_size),
        "--json-output", temp_json_dir,
        "--long-edge", str(long_edge)
    ]
    subprocess.run(command, check=True)

# def concatenate_json_files(temp_json_dir, output_file):
#     json_files = sorted(glob.glob(os.path.join(temp_json_dir, '*.json')))
    
#     with open(output_file, 'w') as f:
#         # Start the JSON object
#         f.write('{"pifpaf": {')
        
#         for frame_index, json_file in enumerate(json_files, start=1):
#             with open(json_file, 'r') as jf:
#                 data = json.load(jf)
#                 frame_data = {
#                     "frame": frame_index,
#                     "predictions": data
#                 }
#                 # Write the frame data
#                 f.write(json.dumps(frame_data))
#                 # Add a comma between frame objects except after the last one
#                 if frame_index < len(json_files):
#                     f.write(',')
        
#         # End the JSON object
#         f.write('}}')
def concatenate_json_files(temp_json_dir, output_csv_file):
    # Get a list of all JSON files in the directory, sorted by name
    json_files = sorted(glob.glob(os.path.join(temp_json_dir, '*.json')))
    
    # Define the CSV headers
    headers = ['Frame', 'Bbox', 'Score', 'Category_ID']
    headers.extend([f'Keypoint_{i+1}' for i in range(17)])  # Adding keypoints

    # Open the CSV file for writing
    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)  # Write the header row

        # Iterate over each JSON file
        for frame_index, json_file in enumerate(json_files, start=1):
            with open(json_file, 'r') as jf:
                data = json.load(jf)
                
                # Iterate over each object (detection) in the JSON file
                for detection in data:
                    bbox = detection.get('bbox', [])
                    score = detection.get('score', 0)
                    category_id = detection.get('category_id', 0)
                    keypoints = detection.get('keypoints', [])

                    # Format keypoints as tuples (x, y, confidence)
                    keypoint_tuples = [f'({keypoints[i]}, {keypoints[i+1]}, {keypoints[i+2]})' 
                                       for i in range(0, len(keypoints), 3)]
                    
                    # Prepare the row
                    row = [
                        frame_index, 
                        str(bbox), 
                        score, 
                        category_id
                    ] + keypoint_tuples
                    
                    # Write the row to the CSV file
                    writer.writerow(row)

def process_images_in_subfolders(input_folder, output_folder):
    parent_dir = input_folder
    output_json_dir = output_folder

    json_file_path = '/scratch-shared/ivanmiert/processed_subfolders.json' 
    # Load subfolders to process from the JSON file
    subfolders_to_use = load_subfolders_from_json(json_file_path)

    # Create the output directory if it doesn't exist
    os.makedirs(output_json_dir, exist_ok=True)

    # Loop over each subfolder in the parent directory
    for subfolder in os.listdir(parent_dir):
        if subfolder not in subfolders_to_use:
            continue

        output_file = os.path.join(output_json_dir, f"{subfolder}.csv")

        # Check if the output file already exists
        if os.path.exists(output_file):
            print(f"Skipping {subfolder}, output already exists.")
            continue

        subfolder_path = os.path.join(parent_dir, subfolder)
        if os.path.isdir(subfolder_path):
            temp_json_dir = os.path.join(output_json_dir, f"temp_{subfolder}")
            os.makedirs(temp_json_dir, exist_ok=True)

            # Run OpenPifPaf on the current subfolder
            run_openpifpaf_on_folder(subfolder_path, temp_json_dir)

            # Concatenate all JSON files in the temp directory into one JSON file
            concatenate_json_files(temp_json_dir, output_file)

            # Remove the temporary directory
            for file in os.listdir(temp_json_dir):
                os.remove(os.path.join(temp_json_dir, file))
            os.rmdir(temp_json_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process subfolders of images with OpenPifPaf and combine outputs into separate JSON files per subfolder")
    parser.add_argument('--input_folder', type=str, required=True, help="Path to the folder containing subfolders with images")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder to save JSON output files")
    args = parser.parse_args()
    
    process_images_in_subfolders(args.input_folder, args.output_folder)