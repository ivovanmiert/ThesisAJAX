import sys
sys.path.insert(0, '/home/ivanmiert/pose-estimation')
from infer_overview import run_pipeline
import yaml
import os 
import json
import math

# Load configuration from YAML file
def load_config(config_path='/home/ivanmiert/overview/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Example usage
config = load_config()  # Assuming config.yaml is in the same directory

def load_json(json_path, current_part):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    # Extract the subfolder names into a list
    subfolders_to_process = [item['subfolder'].split('_')[-1] for item in json_data]
    total_subfolders = len(subfolders_to_process)
    subfolders_per_part = total_subfolders

    # Calculate the start and end indices for the current part
    start_idx = (current_part - 1) * subfolders_per_part
    end_idx = min(current_part * subfolders_per_part, total_subfolders)
    
    # Return the subfolders for the current part
    return subfolders_to_process[start_idx:end_idx]


pose_model = config['model_paths']['top_down_path']
img_size = 1280
conf_thres = config['confidences']['top_down_confidence']
iou_thres = config['confidences']['top_down_iou']

detection_folder_base = config['template_folders']['player_detect']
frames_folder_base = config['template_folders']['base']
output_folder_base = config['template_folders']['top_down_pred']
json_folders_to_process_base = config['template_folders']['keypoints_list']

current_chunk = config['info']['current_chunk']
current_part_top_down = config['info']['current_part_top_down']
print(current_chunk)
print(current_part_top_down)
detections_folder = os.path.join(detection_folder_base, f"chunk_{current_chunk}")
frames_folder = os.path.join(frames_folder_base, f"chunk_{current_chunk}")
output_folder = os.path.join(output_folder_base, f"chunk_{current_chunk}")
json_path = os.path.join(json_folders_to_process_base, f"chunk_{current_chunk}.json")


subfolders_to_process = load_json(json_path, current_part_top_down)
print(subfolders_to_process)

#event_range = [0, 99999999999]

#event_range = [0, 1723218370]
#event_range = [1723218371, 1764449550]
#event_range = [1764449551, 1805680730]
#event_range = [1805680731 , 1846911910]
#event_range = [1846911911 , 1888143090]
#event_range = [1888143091 , 1929374270]
#event_range = [1929374271 , 1970605450]
#event_range = [1970605451 , 2011836630]
#event_range = [2011836631 , 2053067810]
#event_range = [2053067811 , 2094298990]
#event_range = [2094298991 , 9999999999]


# Call the pipeline function
run_pipeline(
    pose_model,
    img_size,
    conf_thres,
    iou_thres,
    detections_folder,
    frames_folder,
    output_folder,
    subfolders_to_process
)