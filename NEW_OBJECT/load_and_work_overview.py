import os
import time
import pandas as pd
from general import EventData
from field import FieldKeypoints, AllFieldKeypoints
from kits import AllKits_Fieldtypes
from event_files import EventFiles

class Events:
    def __init__(self, json_field, excel_kits_field, path_ball_detection_folder, path_player_detection_folder, path_hpe_detection_folder, path_image_per_event_folder, path_keypoints_field_prediction_folder, class_names, classify_model_path, event_df, json_subfolders_to_use, hpe_sort, current_part, path_player_features_folder, path_ball_features_folder):
        self.Field_Types_Keypoints = AllFieldKeypoints(json_field)
        self.All_Kits_and_Field_Type = AllKits_Fieldtypes(excel_kits_field)
        self.Event_Files = EventFiles(path_ball_detection_folder, path_player_detection_folder, path_hpe_detection_folder, path_image_per_event_folder, path_keypoints_field_prediction_folder, json_subfolders_to_use, hpe_sort, current_part)
        self.class_names = class_names
        self.classify_model_path = classify_model_path
        self.event_df = event_df
        self.event_data_objects = {}
        self.events_without_ball_detection = []
        self.hpe_sort = hpe_sort
        self.subfolders_to_process = self.Event_Files.subfolders_to_use
        self.path_player_features_folder = path_player_features_folder
        self.path_ball_features_folder = path_ball_features_folder

    def instantiate_events(self, events):
        start_time = time.time()

        for subfolder in self.subfolders_to_process:
            event = int(subfolder.split('_')[1])
            print('event:')
            print(event)
            if event not in events:
                print('zit hier niet in')
                continue
            # output_file_path = output_path_template.format(event)
            # print(output_file_path)
            # if os.path.exists(output_file_path):
            #     print(f'File for event {event} already exists. Skipping...')
            #     continue  # Skip if the file already exists

            print(f'Starting with event: {event}')
            player_detection = self.Event_Files.get_players_detection(f'{event}')
            hpe_detection = self.Event_Files.get_HPE(f'{event}')
            ball_detection = self.Event_Files.get_ball_detection(f'{event}')
            field_keypoints_prediction = self.Event_Files.get_keypoints_field(f'{event}')
            match_id = self.event_df.loc[self.event_df['event_id'] == event, 'match_id'].values[0]
            print(match_id)
            folder_frames = self.Event_Files.get_images_folder(f'{event}')
            #print(player_detection, hpe_detection, ball_detection, field_keypoints_prediction, match_id, folder_frames)
            event_data = EventData(event, player_detection, hpe_detection, ball_detection, field_keypoints_prediction, self.Field_Types_Keypoints, match_id, folder_frames, self.All_Kits_and_Field_Type, self.classify_model_path, self.class_names, self.hpe_sort, self.path_player_features_folder, self.path_ball_features_folder)
            self.event_data_objects[event] = event_data
            
            # if event_data.ball is None or event_data.homography_estimator.valid_homographies is False:
            #     self.events_without_ball_detection.append(event)

        end_time = time.time()
        print('Events without ball detection:')
        print(self.events_without_ball_detection)
        print(f'Time taken: {end_time - start_time} seconds')

    def get_event_data_object(self, event_id):
        return self.event_data_objects.get(event_id)