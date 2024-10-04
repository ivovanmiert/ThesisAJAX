import os
import pandas as pd
from ultralytics import YOLO
import numpy as np
import cv2
import torch
import json

# Paths
# Paths
image_base_folder = '/scratch-shared/ivanmiert/final_events_folder/frames_10000'  # Update this to your root folder containing folders of frames
output_dir = '/scratch-shared/ivanmiert/final_events_folder/ball_detect'  # YOLO's output directory
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# Initialize the YOLO model
model = YOLO('yolov8m-football.pt')

# List all folders in the root directory
# video_folders = [d for d in os.listdir(frames_root_dir) if os.path.isdir(os.path.join(frames_root_dir, d))]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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
# def remove_invalid_frames(frames_folder_path):
#     """Remove invalid frames with zero bytes."""
#     frame_files = [f for f in os.listdir(frames_folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
#     for frame_file in frame_files:
#         frame_path = os.path.join(frames_folder_path, frame_file)
#         if os.path.getsize(frame_path) == 0:
#             print(f'Removing invalid frame: {frame_path}')
#             os.remove(frame_path)

# Loop through each folder and run YOLO detection on frames
json_file_path = '/scratch-shared/ivanmiert/processed_subfolders.json'
subfolders_to_use = load_subfolders_from_json(json_file_path)

#print(f"Performing on subfolders in base folder: {image_base_folder}")
clip_folders = sorted(os.listdir(image_base_folder))
#print(f"Clip folders found: {clip_folders}")

# Filter clip_folders to include only those in subfolders_to_use
clip_folders = [folder for folder in clip_folders if folder in subfolders_to_use]
#print(f"Filtered clip folders: {clip_folders}")

for video_folder in clip_folders:
    frames_folder_path = os.path.join(image_base_folder, video_folder)
    csv_file = video_folder + '.csv'
    csv_file_path = os.path.join(output_dir, csv_file)
    
    # Check if the CSV file already exists
    if os.path.exists(csv_file_path):
        print(f'{csv_file_path} already exists, skipping...')
        continue
    # Remove invalid frames
    #remove_invalid_frames(frames_folder_path)
    frame_files = sorted([f for f in os.listdir(frames_folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))])

    # Check if frames are valid (non-zero bytes)
    valid_frame_files = [f for f in frame_files if os.path.getsize(os.path.join(frames_folder_path, f)) > 0]
    
    if not valid_frame_files:
        print(f'No valid frames found in {frames_folder_path}, skipping...')
        continue

    print(f"frames_folder_path: {frames_folder_path}")
    
    # # Load all frames from the folder
    # frames = []
    # for frame_file in frame_files:
    #     frame_path = os.path.join(frames_folder_path, frame_file)
    #     frames.append(frame_path)
    print(f"frames_folder_path:{frames_folder_path}")
    # Run YOLO detection
    results = model.predict(frames_folder_path, save=False, conf=0.25, imgsz=1280, line_thickness=1, device=0)
    
    # Aggregate detections for the current video
    all_detections = []

    for frame_number, result in enumerate(results):
        detections = result[:, :4].cpu().numpy()  # Get bbox in xywh format
        confidences = result[:, 4].cpu().numpy()  # Get confidence scores
        classes = result[:, 5].cpu().numpy().astype(int)  # Get class labels
        
        # Combine detections, confidences, and classes into a single array
        frame_detections = np.hstack((classes.reshape(-1, 1), detections, confidences.reshape(-1, 1)))
        
        # Filter detections to only include the ball (class label 0)
        ball_detections = frame_detections[frame_detections[:, 0] == 0]
        
        if ball_detections.size > 0:
            # Append frame number to each detection
            frame_numbers = np.full((ball_detections.shape[0], 1), frame_number)
            ball_detections = np.hstack((frame_numbers, ball_detections))
        
            # Append to all_detections
            all_detections.append(ball_detections)
    
    # Convert the list of arrays to a single array if there are any detections
    if all_detections:
        all_detections = np.vstack(all_detections)
    else:
        all_detections = np.empty((0, 7))  # Create an empty array with 7 columns
    
    # Define the columns for the CSV file
    columns = ['frame_number', 'class', 'x_center', 'y_center', 'width', 'height', 'confidence']
    
    # Save as CSV
    pd.DataFrame(all_detections, columns=columns).to_csv(csv_file_path, index=False)
    
    print(f'Saved {csv_file_path}')

print('Processing complete.')