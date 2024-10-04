import os
import glob
import subprocess
import json
import csv
import time


def load_subfolders_from_json(json_file_path):
    """Load subfolders to process from a JSON file."""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            subfolders = [entry['subfolder'] for entry in data if 'subfolder' in entry]
            return subfolders
    except Exception as e:
        print(f"Error loading or parsing JSON file: {e}")
        return []


def run_openpifpaf_on_folder(input_folder, temp_json_dir, batch_size=1):
    """Run OpenPifPaf on a folder of images."""
    long_edge = 1280
    command = [
        "python3", "-m", "openpifpaf.predict",
        "--glob", f"{input_folder}/*.jpg",
        "--batch-size", str(batch_size),
        "--json-output", temp_json_dir,
        "--long-edge", str(long_edge),
        "--keypoint-threshold", "0.25",
        "--instance-threshold", "0.05",
        #"--force-complete-pose", 
        "--cif-th", "0.05", 
        "--caf-th", "0.05"
    ]
    subprocess.run(command, check=True)


def concatenate_json_files(temp_json_dir, output_csv_file):
    """Concatenate all JSON files in a folder and save to a CSV file."""
    json_files = sorted(glob.glob(os.path.join(temp_json_dir, '*.json')))
    headers = ['Frame', 'Bbox', 'Score', 'Category_ID']
    headers.extend([f'Keypoint_{i+1}' for i in range(17)])

    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for frame_index, json_file in enumerate(json_files, start=1):
            with open(json_file, 'r') as jf:
                data = json.load(jf)
                for detection in data:
                    bbox = detection.get('bbox', [])
                    score = detection.get('score', 0)
                    category_id = detection.get('category_id', 0)
                    keypoints = detection.get('keypoints', [])
                    keypoint_tuples = [f'({keypoints[i]}, {keypoints[i+1]}, {keypoints[i+2]})' for i in range(0, len(keypoints), 3)]
                    row = [frame_index, str(bbox), score, category_id] + keypoint_tuples
                    writer.writerow(row)


def process_images_in_subfolders(input_folder, output_folder):
    """Process images in subfolders, run OpenPifPaf, and save outputs."""
    
    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist.")
        return

    # subfolders_to_use = load_subfolders_from_json(json_file_path)
    # os.makedirs(output_folder, exist_ok=True)

    for subfolder in os.listdir(input_folder):
        # if subfolder not in subfolders_to_use:
        #     print('hello3')
        #     continue

        output_file = os.path.join(output_folder, f"{subfolder}.csv")
        if os.path.exists(output_file):
            print(f"Skipping {subfolder}, output already exists.")
            continue

        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            temp_json_dir = os.path.join(output_folder, f"temp_{subfolder}")
            os.makedirs(temp_json_dir, exist_ok=True)
            start_time = time.time()
            run_openpifpaf_on_folder(subfolder_path, temp_json_dir)
            end_time = time.time()
            print(f"TOTAL TIME = {end_time - start_time}")
            concatenate_json_files(temp_json_dir, output_file)

            # Cleanup
            for file in os.listdir(temp_json_dir):
                os.remove(os.path.join(temp_json_dir, file))
            os.rmdir(temp_json_dir)


