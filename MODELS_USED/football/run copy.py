import os
import pandas as pd
from ultralytics import YOLO
import numpy as np

# Paths
video_dir = 'vid_directory'  # Update this to your folder of videos
output_dir = 'output_directory'  # YOLO's output directory
csv_dir = 'csv_results'
os.makedirs(csv_dir, exist_ok=True)

# Initialize the YOLO model
model = YOLO('yolov8m-football.pt')

# List all video files in the folder
video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

# Loop through each video and run YOLO detection
for video_file in video_files:
    video_path = os.path.join(video_dir, video_file)
    result_dir = os.path.join(output_dir, os.path.splitext(video_file)[0])
    
    # Run YOLO detection
    results = model.predict(source=video_path, save=False, conf=0.25, imgsz=1280, line_thickness=1)
    
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
    csv_file = os.path.splitext(video_file)[0] + '.csv'
    csv_file_path = os.path.join(csv_dir, csv_file)
    pd.DataFrame(all_detections, columns=columns).to_csv(csv_file_path, index=False)
    
    print(f'Saved {csv_file_path}')

print('Processing complete.')