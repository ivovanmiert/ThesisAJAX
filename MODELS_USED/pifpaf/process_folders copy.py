import os
import argparse
import subprocess
import json

def process_images_in_subfolders(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over all subfolders in the input folder
    for subfolder_name in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder_name)
        
        if os.path.isdir(subfolder_path):
            all_results = []
            output_file = os.path.join(output_folder, f"{subfolder_name}.json")
            
            # Iterate over all image files in the subfolder
            for filename in os.listdir(subfolder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_id = os.path.splitext(filename)[0]
                    input_image_path = os.path.join(subfolder_path, filename)
                    print(f"now handling {image_id}")
                    
                    # Call OpenPifPaf to process the image and get JSON output
                    cmd = [
                        "python3", "-m", "openpifpaf.predict ",
                        input_image_path,
                        "--json-output"
                    ]
                    print(cmd)
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    json_output = json.loads(result.stdout)
                    
                    # Add image_id to each entry in the result for identification
                    for item in json_output:
                        item['image_id'] = image_id
                    
                    all_results.extend(json_output)
                    print(f"Processed {filename} in {subfolder_name}")
            
            # Save all results to a single JSON file for the current subfolder
            with open(output_file, 'w') as f:
                json.dump(all_results, f)
            print(f"All results for {subfolder_name} saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process subfolders of images with OpenPifPaf and combine outputs into separate JSON files per subfolder")
    parser.add_argument('--input_folder', type=str, required=True, help="Path to the folder containing subfolders with images")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder to save JSON output files")
    args = parser.parse_args()
    
    process_images_in_subfolders(args.input_folder, args.output_folder)