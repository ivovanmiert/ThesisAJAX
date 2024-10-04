import yaml
import sys 
sys.path.insert(0, '/home/ivanmiert/sportlight_folder/soccernet-calibration-sportlight/src/models/hrnet')

from overview_prediction import main
import os

# Load configuration from YAML file
def load_config(config_path='/home/ivanmiert/overview/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Example usage
config = load_config()  # Assuming config.yaml is in the same directory

model_path = config['model_paths']['calib_path']
frames_folder_base = config['template_folders']['base']
output_base_folder = config['template_folders']['keypoints_pred']
current_chunk = config['info']['current_chunk']
json_folders_to_process_base = config['template_folders']['keypoints_list']


frames_folder = os.path.join(frames_folder_base, f"chunk_{current_chunk}")
output_folder = os.path.join(output_base_folder, f"chunk_{current_chunk}")
json_path = os.path.join(json_folders_to_process_base, f"chunk_{current_chunk}.json")




# Run the main function
main(model_path, frames_folder, output_folder, json_path)
