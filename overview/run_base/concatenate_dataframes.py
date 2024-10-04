import sys 
sys.path.insert(0, '/home/ivanmiert/NEW_OBJECT/CLASSIFICATION_MODEL')

from example_overview import combine_player_ball_data
import yaml
import os

# Load configuration from YAML file
def load_config(config_path='/home/ivanmiert/overview/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Example usage
config = load_config()  # Assuming config.yaml is in the same directory

player_folder_base = config['template_folders']['player_features_bottom_up']
ball_folder_base = config['template_folders']['ball_features_bottom_up']
output_folder = config['template_folders']['concatenated_features_bottom_up']
current_chunk = config['info']['current_chunk']

player_folder = os.path.join(player_folder_base, f"chunk_{current_chunk}")
ball_folder = os.path.join(ball_folder_base, f"chunk_{current_chunk}")

combine_player_ball_data(player_folder, ball_folder, output_folder)