import sys
sys.path.insert(0, '/home/ivanmiert/playerDET/scripts/tracking')
from main_tracking_overview import main
import yaml
import os
import json


# Load configuration from YAML file
def load_config(config_path='/home/ivanmiert/overview/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Example usage
config = load_config()  # Assuming config.yaml is in the same directory

# Load the JSON file containing subfolders to process
def load_json(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    # Extract the subfolder names into a list
    subfolders_to_process = [item['subfolder'] for item in json_data]
    return subfolders_to_process

#output_path_csv = config['test_detections']['output_folder_csv']
#image_folder = config['test_detections']['frames_folder']
output_folder_base = config['template_folders']['player_detect']
frames_folder_base = config['template_folders']['base']
current_chunk = config['info']['current_chunk']
json_folders_to_process_base = config['template_folders']['keypoints_list']


frames_folder = os.path.join(frames_folder_base, f"chunk_{current_chunk}")
output_folder = os.path.join(output_folder_base, f"chunk_{current_chunk}")
json_path = os.path.join(json_folders_to_process_base, f"chunk_{current_chunk}.json")
print(current_chunk)
subfolders_to_process = load_json(json_path)
os.makedirs(output_folder, exist_ok=True)
#event_range = [0, 1723218370]
#event_range = [1723218371, 1764449550]
# event_range = [1764449551, 1805680730]
# event_range = [1805680731 , 1846911910]
# event_range = [1846911911 , 1888143090]
# event_range = [1888143091 , 1929374270]
# event_range = [1929374271 , 1970605450]
# event_range = [1970605451 , 2011836630]
# event_range = [2011836631 , 2053067810]
# event_range = [2053067811 , 2094298990]
# event_range = [2094298991 , 9999999999]

main(output_folder, frames_folder, subfolders_to_process)