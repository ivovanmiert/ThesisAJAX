import sys
sys.path.insert(0, '/home/ivanmiert/pifpaf/test')
from process_folders_overview import process_images_in_subfolders
import yaml
import os

# Load configuration from YAML file
def load_config(config_path='/home/ivanmiert/overview/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Example usage
config = load_config()  # Assuming config.yaml is in the same directory


output_folder_base = config['template_folders']['pifpaf_pred']
frames_folder_base = config['template_folders']['base']
current_chunk = config['info']['current_chunk']
json_folders_to_process_base = config['template_folders']['keypoints_list']
#input_folder = config['test_detections']['frames_folder']
#output_folder = config['test_detections']['output_folder_pifpaf']
frames_folder = os.path.join(frames_folder_base, f"chunk_{current_chunk}")
output_folder = os.path.join(output_folder_base, f"chunk_{current_chunk}")
json_path = os.path.join(json_folders_to_process_base, f"chunk_{current_chunk}.json")
print(current_chunk)
process_images_in_subfolders(frames_folder, output_folder, json_path)

