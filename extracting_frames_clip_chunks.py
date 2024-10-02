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

# Extract frames from a video file
def extract_event_frames(video_path, output_folder, start_frame, end_frame, gpu_id, event_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    event_folder = os.path.join(output_folder, f"event_{event_id}")
    os.makedirs(event_folder, exist_ok=True)
    video_capture = cv2.VideoCapture(video_path)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_number = current_frame - start_frame
        padded_frame_number = f"{frame_number:04d}"
        output_filename = os.path.join(event_folder, f"frame_{padded_frame_number}.jpg")
        cv2.imwrite(output_filename, frame)
        current_frame += 1
    video_capture.release()

# Parallel processing of events
def parallel_event_extraction(df, output_folder, num_gpus, frame_rate):
    tasks = []
    for index, row in df.iterrows():
        match_id = row['match_id']
        event_id = row['event_id']
        video_path = f"/scratch-shared/ivanmiert/All_matches_27aug/g{match_id}-hd.mp4"
        if math.isnan(row['begin_clip_timestamp']) or math.isnan(row['end_clip_timestamp']):
            continue
        start_frame = int(row['begin_clip_timestamp'] * frame_rate)
        end_frame = int(row['end_clip_timestamp'] * frame_rate)
        gpu_id = index % num_gpus
        tasks.append((video_path, output_folder, start_frame, end_frame, gpu_id, event_id))

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(extract_event_frames, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

# Process CSV files for the current chunk
def process_csv_files_for_chunk(chunk_number, config, num_gpus, frame_rate):
    print('hallo')
    chunk_template = config['event_data']['chunk_template']
    csv_path = chunk_template.format(chunk_number)
    
    if not os.path.exists(csv_path):
        print(f"CSV file for chunk {chunk_number} not found.")
        return

    df = pd.read_csv(csv_path)
    output_folder = os.path.join(config['frame_folders']['base'], f"chunk_{chunk_number}")
    print(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    parallel_event_extraction(df, output_folder, num_gpus, frame_rate)

# Main function to process all chunks based on the configuration
config = load_config()
current_chunk = config['info']['current_chunk']
num_gpus = 4
frame_rate = 25

process_csv_files_for_chunk(current_chunk, config, num_gpus, frame_rate)

