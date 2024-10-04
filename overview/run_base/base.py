import sys 
sys.path.insert(0, '/home/ivanmiert/NEW_OBJECT/')
import yaml
import os
import pandas as pd
from load_and_work_overview import Events  
from check_intersection import intersection

"""
This file is used to process the data obtained from the 'run_models' into features used for the classification model. 
Since this process takes about 45 seconds per event, this was done in chunks, so that multiple events could be handled at the same time. 
These chunks were changed in the config file, and then a new job would be sent. 
The data of all models is processed in the functions written in the 'Create Features' folder. 
"""



def load_config(config_path='/home/ivanmiert/overview/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config() 


json_field = config['base']['json_field_coordinates']
excel_kits_field = config['base']['excel_kits_field']
class_names = config['base']['class_names']
classify_model_path = config['base']['classify_model_path']

event_df = pd.read_csv(config['event_data']['all'])

current_chunk = config['info']['current_chunk']
current_part = config['info']['current_part']
hpe_sort = config['info']['hpe_sort']

json_folders_to_process_base = config['template_folders']['keypoints_list']
path_ball_detection_folder_base = config['template_folders']['ball_detect_processed']
path_player_detection_folder_base = config['template_folders']['player_detect']
path_hpe_detection_folder_base = config['template_folders']['top_down_pred']
path_keypoints_field_prediction_folder_base = config['template_folders']['keypoints_pred']
path_image_per_event_folder_base = config['template_folders']['base']
path_player_features_base = config['template_folders']['player_features_top_down']
path_ball_features_base = config['template_folders']['ball_features_top_down']

json_path = os.path.join(json_folders_to_process_base, f"chunk_{current_chunk}.json")
path_ball_detection_folder = os.path.join(path_ball_detection_folder_base, f"chunk_{current_chunk}")
path_player_detection_folder = os.path.join(path_player_detection_folder_base, f"chunk_{current_chunk}")
path_hpe_detection_folder = os.path.join(path_hpe_detection_folder_base, f"chunk_{current_chunk}")
path_keypoints_field_prediction_folder = os.path.join(path_keypoints_field_prediction_folder_base, f"chunk_{current_chunk}")
path_image_per_event_folder = os.path.join(path_image_per_event_folder_base, f"chunk_{current_chunk}")
path_player_features_folder = os.path.join(path_player_features_base, f"chunk_{current_chunk}")
path_ball_features_folder = os.path.join(path_ball_features_base, f"chunk_{current_chunk}")

json_path = os.path.join(json_folders_to_process_base, f"chunk_{current_chunk}.json")

events = intersection(current_chunk)

# Instantiate the Events class with the provided input paths and data
events_object = Events(json_field, excel_kits_field, path_ball_detection_folder, path_player_detection_folder, path_hpe_detection_folder, path_image_per_event_folder, path_keypoints_field_prediction_folder, class_names, classify_model_path, event_df, json_path, hpe_sort, current_part, path_player_features_folder, path_ball_features_folder)

events_object.instantiate_events(events)
