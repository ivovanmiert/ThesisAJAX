import os
import yaml
import cv2
import concurrent.futures
import pandas as pd
import math

# Load configuration from YAML file
def load_config(config_path='/home/ivanmiert/overview/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Example usage
config = load_config()  # Assuming config.yaml is in the same directory

def extract_event_frames(video_path, output_folder, start_frame, end_frame, gpu_id, event_id):
    # Set the environment variable for GPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Create output directory if it doesn't exist
    event_folder = os.path.join(output_folder, f"event_{event_id}")
    os.makedirs(event_folder, exist_ok=True)
    print(video_path)
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Set the video to start at the start_frame
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    print(start_frame)

    current_frame = start_frame

    while current_frame <= end_frame:
        # Read the frame at the current position
        ret, frame = video_capture.read()
        print(current_frame)
        # Break the loop if no frame is returned (end of video)
        if not ret:
            print(f"End of video reached at frame {current_frame}")
            break

        # Calculate zero-padded frame number with 4 digits
        frame_number = current_frame - start_frame
        padded_frame_number = f"{frame_number:04d}"

        # Save the frame as an image file
        output_filename = os.path.join(event_folder, f"frame_{padded_frame_number}.jpg")
        cv2.imwrite(output_filename, frame)
        print(f"Saved {output_filename}")

        current_frame += 1

    # Release the video capture object
    video_capture.release()
    print(f"Frame extraction complete for event {event_id} on GPU {gpu_id}.")


def parallel_event_extraction(df, output_folder, num_gpus, frame_rate):
    # Create a list of tasks for each GPU
    tasks = []
    for index, row in df.iterrows():
        match_id = row['match_id']
        event_id = row['event_id']
        video_path = f"/scratch-shared/ivanmiert/All_matches_27aug/g{match_id}-hd.mp4"
        
        # Check if timestamps are NaN and skip if they are
        if math.isnan(row['begin_clip_timestamp']) or math.isnan(row['end_clip_timestamp']):
            print(f"Skipping event {event_id} due to NaN timestamp.")
            continue
        # Convert timestamps to frame numbers
        start_frame = int(row['begin_clip_timestamp'] * frame_rate)
        end_frame = int(row['end_clip_timestamp'] * frame_rate)
        #event_folder = os.path.join(output_folder, f"event_{event_id}")
        print(f"start_frame: {start_frame}")
        print(f"end_frame: {end_frame}")
        # Check if the output folder already exists
        # if os.path.exists(event_folder):
        #     print(f"Skipping event {event_id} as the output folder already exists.")
        #     continue  # Skip this event and move to the next one

        gpu_id = index % num_gpus  # Distribute tasks across available GPUs
        tasks.append((video_path, output_folder, start_frame, end_frame, gpu_id, event_id))

    # Execute the tasks in parallel using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(extract_event_frames, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")


def process_csv_files(folder_path, output_folder, num_gpus, frame_rate, file_indices=None):
    # Get the list of CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    csv_files.sort()  # Sort files to make the indexing consistent
    
    # If file_indices are provided, select only those files
    if file_indices:
        selected_files = [csv_files[i] for i in file_indices if i < len(csv_files)]
    else:
        selected_files = csv_files  # Process all files if no specific indices provided

    for file in selected_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        # Perform your operations on the dataframe
        print(f"Processing {file_path}")
        parallel_event_extraction(df, output_folder, num_gpus, frame_rate)
        # You can add more code to process each DataFrame here



# Usage example:
main_folder = config['folders']['main_folder']
# Create the output folder if it doesn't exist
os.makedirs(main_folder, exist_ok=True)
folder_path = config['folders']['dataframes_chunks_folder']
output_folder = config['folders']['output_folder_frames']
# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
# If you want to process only the second and third files
num_gpus = 4
frame_rate = 25

process_csv_files(folder_path, output_folder, num_gpus, frame_rate, file_indices=[1])