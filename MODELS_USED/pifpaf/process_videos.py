import os
import argparse
import subprocess

def process_videos(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            event_id = os.path.splitext(filename)[0]
            input_video_path = os.path.join(input_folder, filename)
            output_json_path = os.path.join(output_folder, f"{event_id}.json")
            
            # Call OpenPifPaf to process the video and output JSON
            cmd = [
                "python3", "-m", "openpifpaf.video",
                "--source", input_video_path,
                "--json-output", output_json_path
            ]
            subprocess.run(cmd, check=True)
            print(f"Processed {filename}, JSON saved to {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a folder of videos with OpenPifPaf")
    parser.add_argument('--input_folder', type=str, required=True, help="Path to the folder containing input videos")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder to save JSON outputs")
    args = parser.parse_args()
    
    process_videos(args.input_folder, args.output_folder)