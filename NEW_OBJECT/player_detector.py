import numpy as np
import pandas as pd
import cv2

"""
This class is regarding the transformation of the player detection outcomes, into easy accessible detections for other functions in this folder
"""

class Player_Detector:
    def __init__(self, file_path):
        self.path_to_csv = file_path

    
    def process_csv(file_path):
        df = pd.read_csv(file_path, header=None, names=['frame_id', 'track_id', 'x_min', 'y_min', 'x_max', 'y_max', '-1', '-2', '-3', '-4'])
        grouped = df.groupby('frame_id')
    
        def transform_row(group):
            return [
                [row['track_id'], [row['x_min'], row['y_min'], row['x_max'], row['y_max']]]
                for idx, row in group.iterrows()
            ]
        
        transformed_data = grouped.apply(transform_row)
        processed_df = pd.DataFrame({
            'frame_id': transformed_data.index,
            'player_detections': transformed_data.values
        }).reset_index(drop=True)
        
        return processed_df
        
