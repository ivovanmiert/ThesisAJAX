import csv 
import numpy as np
import cv2
import pandas as pd
import time
import json

pd.set_option('display.max_columns', 500)

from utils import load_detection_df
from utils import calculate_planar_coordinates
from utils import load_ball_df
from utils import calculate_planar_coordinates_ball

from players import Players
from player import Player
from ball import Ball
from homography import HomographyEstimator
from frames import FramesObject
from classify import TeamClassifier
from classify_model import ClassifyModel
from HPE_match import HPE_Matcher
from feature_ball_related import BallInfo
from feature_player_related import PlayerInfo
from featurelist import FeatureListGeneral
import os


"""
This class can be seen as the main class of an Event. From this class, everything about an event is handled. 
"""

class EventData:
    def __init__(self, event_id, player_csv, hpe_csv, ball_csv, field_keypoints_prediction, all_field_keypoints, game_id, folder_frames, all_kits, classify_model, class_names, hpe_sort, chunk_base_player_features, chunk_base_ball_features):
        file_path_ball = os.path.join(chunk_base_ball_features, f'ball_data_{event_id}.csv')
            # Check if the file already exists; if so, skip the whole init
        print(f"file_path_ball: {file_path_ball}")
        if os.path.exists(file_path_ball):
            print(f"Ball data for event {event_id} already exists. Skipping initialization.")
            return  # Skip the rest of the __init__ if the file already exists
        self.event_id = event_id
        start_time_players = time.time()
        self.players = Players()
        end_time_players = time.time()
        time_players = end_time_players - start_time_players
        self.all_field_keypoints = all_field_keypoints
        self.kits = all_kits.get_kits_and_field_type(game_id)
        self.field_keypoints = all_field_keypoints.get_object_certain_type(self.kits['field_type'])
        start_time_classifier_model =time.time()
        self.classifier_model = ClassifyModel(classify_model, class_names)
        end_time_classifier_model = time.time()
        time_classifier_model = end_time_classifier_model - start_time_classifier_model
        start_time_homography_estimator = time.time()
        self.homography_estimator = HomographyEstimator(self.field_keypoints, field_keypoints_prediction)
        end_time_homography_estimator = time.time()
        time_homography_estimator = end_time_homography_estimator - start_time_homography_estimator
        if not self.homography_estimator.valid_homographies:
            print(f"No valid homographies for event {event_id}. Skipping event.")
            return  # Skip the rest of the processing if no valid homographies were found
        start_time_frames_object = time.time()
        self.frames_object = FramesObject(folder_frames)
        end_time_frames_object = time.time()
        time_frames_object = end_time_frames_object - start_time_frames_object
        save_dir = '/home/ivanmiert/NEW_OBJECT/files_folder'
        start_time_team_classifier = time.time()
        self.team_classifier = TeamClassifier(self.kits, self.classifier_model, save_dir)
        end_time_team_classifier = time.time()
        time_team_classifier = end_time_team_classifier - start_time_team_classifier
        start_time_hpe_matcher = time.time()
        self.hpe_matcher = HPE_Matcher(hpe_csv, hpe_sort)
        end_time_hpe_mathcer = time.time()
        time_hpe_matcher = end_time_hpe_mathcer - start_time_hpe_matcher
        self.ball = None
        start_time_load_csv_data = time.time()
        self.load_csv_data(player_csv, ball_csv, hpe_sort)
        if self.ball != None:
            end_time_load_csv_data = time.time()
            time_load_csv = end_time_load_csv_data - start_time_load_csv_data
            self.players.remove_nan_players()
            self.players.remove_players_outside_field(self.field_keypoints)

            self.players.merge_players()
            start_time_feature_list_general = time.time()
            self.feature_list_general = FeatureListGeneral(self.ball, self.field_keypoints, self.players, max_players=100)
            end_time_feature_list_general = time.time()
            time_feature_list = end_time_feature_list_general - start_time_feature_list_general
            dataframe = self.feature_list_general.get_all_player_dataframes()
            dataframe_ball = self.feature_list_general.get_dataframe_ball()

            if not os.path.exists(chunk_base_player_features):
                os.makedirs(chunk_base_player_features)

            file_path_players = os.path.join(chunk_base_player_features, f'players_data_{event_id}.csv')
            dataframe.to_csv(file_path_players, index=False)

            if not os.path.exists(chunk_base_ball_features):
                os.makedirs(chunk_base_ball_features)


            file_path_ball = os.path.join(chunk_base_ball_features, f'ball_data_{event_id}.csv')
            dataframe_ball.to_csv(file_path_ball, index=False)
            print(f"time players: {time_players}, time classifier_model: {time_classifier_model}, time homography estimator: {time_homography_estimator}, time frames object: {time_frames_object}, time team_classifier: {time_team_classifier}, time_hpe matcher: {time_hpe_matcher}, time load csv: {time_load_csv}, time feature list: {time_feature_list}")
        else: 
            print(f'No ball detected in {event_id} so skip:')
        
    def is_similar_bbox(self, bbox1, bbox2, iou_threshold=0.02):
        """
        Check if two bounding boxes are similar based on Intersection over Union (IoU).

        :param bbox1: First bounding box.
        :param bbox2: Second bounding box.
        :param iou_threshold: IoU threshold for considering boxes similar.
        :return: True if the boxes are similar, False otherwise.
        """
        x_min1, y_min1, x_max1, y_max1 = bbox1
        x_min2, y_min2, x_max2, y_max2 = bbox2

        # Calculate intersection
        x_min_inter = max(x_min1, x_min2)
        y_min_inter = max(y_min1, y_min2)
        x_max_inter = min(x_max1, x_max2)
        y_max_inter = min(y_max1, y_max2)

        if x_min_inter < x_max_inter and y_min_inter < y_max_inter:
            intersection_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
        else:
            intersection_area = 0

        bbox1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
        bbox2_area = (x_max2 - x_min2) * (y_max2 - y_min2)

        union_area = bbox1_area + bbox2_area - intersection_area
        iou = intersection_area / union_area if union_area > 0 else 0

        return iou >= iou_threshold
    
    def process_unassigned_hpe_detections(self, hpe_df_unassigned):
        """
        Process unassigned HPE detections to create or update Player objects.

        :param hpe_df_unassigned: DataFrame containing unassigned HPE detections.
        """
        grouped_detections = []

        # Sort detections by frame number for easier processing
        hpe_df_unassigned = hpe_df_unassigned.sort_values(by='Frame')

        # Iterate through unassigned HPE detections
        for index, row in hpe_df_unassigned.iterrows():
            frame_number = int(row['Frame'])
            pixel_bounding_box = row['Bbox']  # Already a list, use directly
            pixel_bounding_box = [max(0, coord) for coord in pixel_bounding_box] #Since some values in the pifpaf have negative pixel values (these are extrapolated) these are changed to 0, since converting them into an image for team classification will error
            xmin, ymin, width, height = pixel_bounding_box
            xmax = xmin + width
            ymax = ymin + height
            pixel_bounding_box = (xmin, ymin, xmax, ymax)
            hpe_coordinates = []
            for i in range(1, 18):
                coord_str = row[f'Keypoint_{i}']
                if isinstance(coord_str, str):
                    coord_tuple = tuple(map(float, coord_str.strip('()').split(',')))
                    hpe_coordinates.append(coord_tuple)
                else:
                    hpe_coordinates.append(coord_str)
            hpe_id = int(row['HPE_ID'])
            
            # Determine if this detection should be grouped with an existing player
            added_to_existing_group = False
            for group in grouped_detections:
                last_bbox = group[-1]['bbox']
                last_frame = group[-1]['frame']

                # Calculate bounding box overlap (IoU) and frame difference
                if self.is_similar_bbox(last_bbox, pixel_bounding_box) and (frame_number - last_frame) <= 5:
                    group.append({
                        'frame': frame_number,
                        'bbox': pixel_bounding_box,
                        'hpe_coordinates': hpe_coordinates,
                        'hpe_id': hpe_id
                    })
                    added_to_existing_group = True
                    break

            # If not similar to any existing group, create a new group
            if not added_to_existing_group:
                grouped_detections.append([{
                    'frame': frame_number,
                    'bbox': pixel_bounding_box,
                    'hpe_coordinates': hpe_coordinates,
                    'hpe_id': hpe_id
                }])

        # Process each group of detections
        for group in grouped_detections:
            first_detection = group[0]
            frame_number = first_detection['frame']-1 #The -1 since the framenumbers of the pifpaf hpe detections start at frame 1 instead of 0
            pixel_bounding_box = first_detection['bbox']
            hpe_coordinates = first_detection['hpe_coordinates']
            hpe_id = first_detection['hpe_id']
            # Calculate pitch coordinates from pixel coordinates
            pitch_coordinates = calculate_planar_coordinates(pixel_bounding_box, frame_number, self.homography_estimator, hpe_id)
            
            # Create the Player object using the first detection
            player_id = self.players.get_highest_player_id() + 1
            team = self.team_classifier.classify(pixel_bounding_box, frame_number, self.frames_object, player_id)
            player = Player(player_id, team, pixel_coordinates=pixel_bounding_box,
                            pitch_coordinates=pitch_coordinates, hpe_pixel_coordinates=hpe_coordinates,
                            current_frame_number=frame_number, hpe_only=True)
            self.players.add_player(player)

            for detection in group[1:]:
                frame_number = detection['frame'] - 1 #The -1 since the framenumbers of the pifpaf hpe detections start at frame 1 instead of 0
                pixel_bounding_box = detection['bbox']
                hpe_coordinates = detection['hpe_coordinates']
                pitch_coordinates = calculate_planar_coordinates(pixel_bounding_box, frame_number, self.homography_estimator, hpe_id)
                self.players.update_player_position(player_id, frame_number, pixel_bounding_box, pitch_coordinates, hpe_coordinates)


    def load_csv_data(self, player_csv, ball_csv, hpe_sort):
        self.load_ball_data(ball_csv)
        self.load_player_data(player_csv, hpe_sort)


    def load_player_data(self, file_path, hpe_sort):
        df = load_detection_df(file_path)
        if hpe_sort == 'bottom_up':
            for index, row in df.iterrows():
                frame_number = int(row['frame_id'])
                player_detections = row['player_detections']

                for detection in player_detections:
                    player_id = detection[0]
                    x_min, y_min, x_max, y_max = detection[1]
                    pixel_bounding_box = (x_min, y_min, x_max, y_max)
                    pitch_coordinates = calculate_planar_coordinates(pixel_bounding_box, frame_number, self.homography_estimator, player_id)
                    

                    hpe_matches = self.hpe_matcher.match_single_detection(frame_number, pixel_bounding_box)  
                    
                    hpe_coordinates, hpe_id, current_iou = None, None, 0
                    for hpe_coordinates, hpe_id, current_iou in hpe_matches:
                        if hpe_coordinates is not None:
                            assigned, assigned_iou = self.hpe_matcher.is_hpe_assigned(hpe_id)
                            if not assigned or current_iou > assigned_iou:
                                if assigned:
                                    previous_player_id = self.hpe_matcher.matched_hpe[hpe_id]['player_id']
                                    self.hpe_matcher.unassign_hpe(hpe_id)
                                    self.players.update_player_hpe_assignment(previous_player_id, frame_number, self.hpe_matcher) 
                                self.hpe_matcher.assign_hpe(hpe_id, player_id, current_iou)
                                break 
            
                    
                    if player_id not in self.players.players_dict:
                        team = self.team_classifier.classify(pixel_bounding_box, frame_number, self.frames_object, player_id)
                        player = Player(player_id, team, pixel_bounding_box, pitch_coordinates, hpe_coordinates, frame_number, hpe_only=False)
                        self.players.add_player(player)
                    else:
                        self.players.update_player_position(player_id, frame_number, pixel_bounding_box, pitch_coordinates, hpe_coordinates)
            highest_ID = self.players.get_highest_player_id()
            # Retrieve the updated HPE DataFrame
            hpe_df = self.hpe_matcher.get_hpe_df()
            hpe_df_unassigned = hpe_df[hpe_df['Assigned'] == 0]
            self.process_unassigned_hpe_detections(hpe_df_unassigned)

        if hpe_sort == 'top_down':
            for index, row in df.iterrows():
                frame_number = int(row['frame_id'])
                player_detections = row['player_detections']


                for detection in player_detections:
                    player_id = detection[0]
                    x_min, y_min, x_max, y_max = detection[1]
                    pixel_bounding_box = (x_min, y_min, x_max, y_max)
                    pitch_coordinates = calculate_planar_coordinates(pixel_bounding_box, frame_number, self.homography_estimator, player_id)
                    hpe_coordinates = self.hpe_matcher.match_detection_top_down(frame_number, player_id)

                    if player_id not in self.players.players_dict:
                        team = self.team_classifier.classify(pixel_bounding_box, frame_number, self.frames_object, player_id)
                        #Here the team classification has to be done. #The classiying classes have been made. Only have to decide on what the best way is to insert the image. Maybe create an Image/Frames class/object with all frames in it. Can easily be fed into the left part of the classifying model + good to subtract bounding boxes for classifying.
                        player = Player(player_id, team, pixel_bounding_box, pitch_coordinates, hpe_coordinates, frame_number, hpe_only=False)
                        self.players.add_player(player)
                    else:
                        self.players.update_player_position(player_id, frame_number, pixel_bounding_box, pitch_coordinates, hpe_coordinates)


    def load_ball_data(self, file_path):
        """
        First the interpolation process was done inside here, but this changed in the final stages of the project, so some code is redundant here, but kept it in but it will just not be called. 
        
        """
        df = load_ball_df(file_path)
        #print(df)
        middle_pixel = (640, 360)
        
        for index, row in df.iterrows():
            frame_id = row['frame_number']
            pixel_coordinates = (row['x_center'], row['y_center'])
            frame_id = int(frame_id)
            pitch_coordinates = calculate_planar_coordinates_ball(pixel_coordinates, frame_id, self.homography_estimator)

            if self.ball is None:
                # Initialization of ball object with the first detection
                self.ball = Ball(pixel_coordinates, pitch_coordinates, frame_id)
            else:
                if self.ball.current_frame_number == frame_id:
                    # If multiple detections in the first frame, compare to middle pixel
                    if len(self.ball.previous_positions) == 0:  # First detection scenario
                        current_distance = (self.ball.pixel_coordinates[0] - middle_pixel[0])**2 + (self.ball.pixel_coordinates[1] - middle_pixel[1])**2
                        new_distance = (pixel_coordinates[0] - middle_pixel[0])**2 + (pixel_coordinates[1] - middle_pixel[1])**2
                    else:
                        previous_pixel_coordinates = self.ball.get_previous_detection()
                        current_distance = (self.ball.pixel_coordinates[0] - previous_pixel_coordinates[0])**2 + (self.ball.pixel_coordinates[1] - previous_pixel_coordinates[1])**2
                        new_distance = (pixel_coordinates[0] - previous_pixel_coordinates[0])**2 + (pixel_coordinates[1] - previous_pixel_coordinates[1])**2
                    
                    if new_distance < current_distance:
                        self.ball.update_position(frame_id, pixel_coordinates, pitch_coordinates, overwrite=True)
                else:
                    # Calculate Euclidean distance between the new detection and the last detection
                    previous_pixel_coordinates = self.ball.pixel_coordinates
                    previous_frame_id = self.ball.current_frame_number
                    frames_since_last_detection = frame_id - previous_frame_id
                    distance = ((pixel_coordinates[0] - previous_pixel_coordinates[0]) ** 2 +
                                (pixel_coordinates[1] - previous_pixel_coordinates[1]) ** 2) ** 0.5
                    max_distance = 100 + 10 * frames_since_last_detection
                    if distance <= max_distance:
                        self.ball.update_position(frame_id, pixel_coordinates, pitch_coordinates)
                    else:
                        print(f"Detection at frame {frame_id} skipped due to large distance: {distance}")
        # After processing all frames, interpolate positions for frames with no detections
    # After processing all frames, check if there is a ball object before interpolating
        if self.ball:
            interpolated_positions = self.ball.interpolate_positions()
            print(f"Interpolated positions: {interpolated_positions}")
        else:
            print("No ball detected, skipping interpolation.")


    def get_ball_object(self):
        return self.ball
    
    def create_feature_list_ball(self):
        feature_list_ball = BallInfo(self.ball, self.field_keypoints, self.players)
        return feature_list_ball.get_features_for_lstm()
    
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
    




    





        

