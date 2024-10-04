import yaml
from post_process_raw import process_csv_files
import os

# Load configuration from YAML file
def load_config(config_path='/home/ivanmiert/overview/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Example usage
config = load_config()  # Assuming config.yaml is in the same directory

#csv_raw_folder_path = config['folders']['output_folder_detection_csv_raw']
#frames_folder_base = config['folders']['output_folder_frames']
#output_folder_csv_processed = config['folders']['output_folder_detection_csv_processed']

csv_raw_folder_path_base = config['template_folders']['ball_detect']
frames_folder_base = config['template_folders']['base']
output_folder_csv_processed_base = config['template_folders']['ball_detect_processed']
current_chunk = config['info']['current_chunk']
print(current_chunk)
csv_raw_folder_path = os.path.join(csv_raw_folder_path_base, f"chunk_{current_chunk}")
frames_folder = os.path.join(frames_folder_base, f"chunk_{current_chunk}")
output_folder = os.path.join(output_folder_csv_processed_base, f"chunk_{current_chunk}")

process_csv_files(csv_raw_folder_path, frames_folder, output_folder)