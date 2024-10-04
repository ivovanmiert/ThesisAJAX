import sys 
sys.path.insert(0, '/home/ivanmiert/NEW_OBJECT/CLASSIFICATION_MODEL')


"""
This file was used to concatenate the player features with the ball features, for both top-down as bottom-up. 
It points to a function called combine_player_ball_data which handles this the right way. 
The corresponding right paths for top-down/bottom-up were defined in the configuration file.
The current state is combining the features for bottom-up

"""

from example_overview import combine_player_ball_data
import yaml

def load_config(config_path='/home/ivanmiert/overview/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


config = load_config()  # Assuming config.yaml is in the same directory

player_folder = config['folders']['player_features_bottom_up']
ball_folder = config['folders']['ball_features_bottom_up']
output_folder = config['folders']['concatenated_features_bottom_up']

combine_player_ball_data(player_folder, ball_folder, output_folder)