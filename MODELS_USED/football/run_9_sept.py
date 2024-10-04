import os
import pandas as pd
from ultralytics import YOLO
import numpy as np
import cv2
import torch
import json

# Paths
# # Paths
# image_base_folder = '/scratch-shared/ivanmiert/final_events_folder/frames_10000'  # Update this to your root folder containing folders of frames
# output_dir = '/scratch-shared/ivanmiert/final_events_folder/ball_detect'  # YOLO's output directory
# if not os.path.isdir(output_dir):
#     os.mkdir(output_dir)

# Initialize the YOLO model
model = YOLO('yolov8m-football.pt')

# List all folders in the root directory
# video_folders = [d for d in os.listdir(frames_root_dir) if os.path.isdir(os.path.join(frames_root_dir, d))]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

frames_folder_main = os.path.abspath('/scratch-shared/ivanmiert/final_events_folder_9sept/')
output_folder = '/home/ivanmiert/YOLOv8-football/football/csv_9sept' 
confidence = 0.1

for video_folder in os.listdir(frames_folder_main):
    frames_folder_path = os.path.join(frames_folder_main, video_folder)
    csv_file = video_folder + '01.csv'
    csv_file_path = os.path.join(output_folder, csv_file)
    # Check if the CSV file already exists
    if os.path.exists(csv_file_path):
        print(f'{csv_file_path} already exists, skipping...')
        continue
    
    print(f"frames_folder_path:{frames_folder_path}")
    # Run YOLO detection
    #confidence = 0.1
    results = model.predict(frames_folder_path, save=False, conf=confidence, imgsz=1280, line_thickness=1, device=0)

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