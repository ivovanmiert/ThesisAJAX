#This file produces a starting dictionary with the features of every starting player from 4 different point of views:
    #1. Voorkant 
    #2. Achterkant (met rugnummer)
    #3. Links
    #4. Rechts


import numpy as np
import cv2
from player_detection import detect_players  # Assuming you have a function for player detection
from feature_extraction import extract_features  # Assuming you have a function for feature extraction


class Create_Features:
    def __init__(self, input_frames, detections, jersey_colors, jersey_numbers):
        self.input_frames = input_frames
        self.detections = detections
        self.jersey_colors = jersey_colors
        self.jersey_numbers = jersey_numbers
        


class PlayerTracker:
    def __init__(self, player_detection_func, feature_extraction_func):
        self.player_detection_func = player_detection_func
        self.feature_extraction_func = feature_extraction_func
        self.prev_bboxes = None

    def determine_orientation(self, prev_bbox, curr_bbox):
        prev_center = np.array([(prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2])
        curr_center = np.array([(curr_bbox[0] + curr_bbox[2]) / 2, (curr_bbox[1] + curr_bbox[3]) / 2])

        displacement = curr_center - prev_center
        displacement_ratio = abs(displacement[0] / displacement[1])

        if displacement_ratio < 0.5:  # Mostly vertical movement
            if displacement[1] > 0:  # Player moving downwards
                return 'front'
            else:  # Player moving upwards
                return 'back'
        elif displacement[0] > 0:  # Player moving towards right
            return 'right'
        else:  # Player moving towards left
            return 'left'

    def extract_player_features(self, frame, bbox, orientation):
        x_min, y_min, x_max, y_max = bbox

        if orientation == 'front':
            player_face = frame[y_min:y_max, x_min:x_max]  # Frontside
        elif orientation == 'back':
            player_back = frame[y_min:y_max, x_min:x_max]  # Backside with jersey number
        elif orientation == 'left':
            player_left = frame[y_min:y_max, x_min:x_max]  # Left side
        elif orientation == 'right':
            player_right = frame[y_min:y_max, x_min:x_max]  # Right side

        # Extract features based on orientation
        if orientation == 'front':
            features = self.feature_extraction_func(player_face)
        elif orientation == 'back':
            features = self.feature_extraction_func(player_back)
        elif orientation == 'left':
            features = self.feature_extraction_func(player_left)
        elif orientation == 'right':
            features = self.feature_extraction_func(player_right)

        return features

    def track_players(self, frames):
        player_features = {}

        for frame_idx, frame in enumerate(frames):
            player_bboxes = self.player_detection_func(frame)

            if self.prev_bboxes is not None:
                for prev_bbox, curr_bbox in zip(self.prev_bboxes, player_bboxes):
                    orientation = self.determine_orientation(prev_bbox, curr_bbox)
                    features = self.extract_player_features(frame, curr_bbox, orientation)

                    player_id = f'player_{len(player_features) + 1}'
                    player_features[player_id] = features

            self.prev_bboxes = player_bboxes

        return player_features