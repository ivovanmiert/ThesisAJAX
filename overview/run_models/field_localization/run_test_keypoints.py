import yaml
import sys 
sys.path.insert(0, '/home/ivanmiert/sportlight_folder/soccernet-calibration-sportlight/src/models/hrnet')

from prediction_mine2_preprocess5_overview import main
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
output_list_base = config['template_folders']['keypoints_list']
current_chunk = config['info']['current_chunk']
print(current_chunk)
frames_folder = os.path.join(frames_folder_base, f"chunk_{current_chunk}")
output_list = os.path.join(output_list_base, f"chunk_{current_chunk}.json")
print(output_list)
# Run the main function
main(model_path, frames_folder, output_list)
