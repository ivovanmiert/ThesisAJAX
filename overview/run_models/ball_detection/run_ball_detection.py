
import torch
import sys 
sys.path.insert(0, '/home/ivanmiert/YOLOv8-football/football')

from final import detect_ball_in_frames
from ultralytics import YOLO
import yaml
import os

# Load configuration from YAML file
def load_config(config_path='/home/ivanmiert/overview/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Example usage
config = load_config()  # Assuming config.yaml is in the same directory

model_path = config['model_paths']['yolo_path']
frames_folder_base = config['template_folders']['base']
current_chunk = config['info']['current_chunk']
output_folder_base = config['template_folders']['ball_detect']
confidence = config['confidences']['ball_detect_confidence']

frames_folder = os.path.join(frames_folder_base, f"chunk_{current_chunk}")
output_folder = os.path.join(output_folder_base, f"chunk_{current_chunk}")
# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

#event_range = [1846911911 , 1888143090]
#event_range = [1888143091 , 1929374270]
#event_range = [1929374271 , 1970605450]
#event_range = [1970605451 , 2011836630]
#event_range = [2011836631 , 2053067810]
#event_range = [2053067811 , 2094298990]
#event_range = [2094298991 , 9999999999]

#frames_folder_main = config['test_detections']['frames_folder']
#output_folder = config['test_detections']['output_ball_detections_test_raw']

# Initialize YOLO model
model = YOLO(model_path)

# Set parameters
# frames_folder_main = '/scratch-shared/ivanmiert/overview/frames_folder'
# output_folder = '/scratch-shared/ivanmiert/overview/ball_detection' 
# confidence = 0.1

# Detect ball in frames using the imported function
detect_ball_in_frames(model, frames_folder, output_folder, confidence)