import os
import json
import math

"""
This class' goal is to get all the paths of the different information sources (ball detection, player detection, hpe detection (bottom-up), images, field localization) and store them
Besides that it loads the different subfolders to use (a list containing the names of the subfolders to use). This list was originally started when there was a folder with frames of 10.000 events and not all events
were processed yet. Now, for the final model structure, there are 10 subfolders which are used subsequently, and this subfolder_to_use list is a little redundant now, but just used to point to the events per 1000.

"""

class EventFiles:
    def __init__(self, ball_detection_dir, players_detection_dir, hpe_detection_dir, images_base_dir, keypoints_field_dir, json_subfolders_to_use, hpe_sort, current_part):
        """
        Initializes the EventFiles object.

        :ball_detection_dir: Directory containing all ball detection CSV files.
        :players_detection_dir: Directory containing all players detection CSV files.
        :hpe_detection_dir: Directory containing all HPE detection CSV files.
        :images_base_dir: Base directory where folders of images for each event are located.
        :keypoints_field_dir: Directory containing all keypoints field .pth files.
        """
        self.ball_detection_dir = ball_detection_dir
        self.players_detection_dir = players_detection_dir
        self.hpe_detection_dir = hpe_detection_dir
        self.images_base_dir = images_base_dir
        self.keypoints_field_dir = keypoints_field_dir
        self.subfolders_to_use = self.load_subfolders_from_json(json_subfolders_to_use, current_part)
        self.hpe_sort = hpe_sort
        self.events = self._load_events()


    def load_subfolders_from_json(self, json_file_path, current_part):
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                
                subfolders = [entry['subfolder'] for entry in data if 'subfolder' in entry]
                
                total_subfolders = len(subfolders)
                subfolders_per_part = math.ceil(total_subfolders / 5)
                
                start_idx = (current_part - 1) * subfolders_per_part
                end_idx = min(current_part * subfolders_per_part, total_subfolders)
                
                return subfolders[start_idx:end_idx]
        
        except Exception as e:
            print(f"Error loading or parsing JSON file: {e}")
            return []

    def _load_events(self):
        """
        Loads all event files from the directories.
        """
        events = {}
        
        for file_name in os.listdir(self.ball_detection_dir):
            if file_name.endswith('.csv'):
                event_id = os.path.splitext(file_name)[0]
                event_id = event_id.split("_")[1]
                events.setdefault(event_id, {})['ball_detection'] = os.path.join(self.ball_detection_dir, file_name)


        for subfolder_name in os.listdir(self.players_detection_dir):
            subfolder_path = os.path.join(self.players_detection_dir, subfolder_name)
            
            if os.path.isdir(subfolder_path): 
                event_id = subfolder_name  
                out_csv_path = os.path.join(subfolder_path, 'out.csv')
                event_id = event_id.split("_")[1]
                if os.path.exists(out_csv_path): 
                    events.setdefault(event_id, {})['players_detection'] = out_csv_path

        if self.hpe_sort == 'bottom_up':
            for file_name in os.listdir(self.hpe_detection_dir):
                if file_name.endswith('.csv'):
                    event_id = os.path.splitext(file_name)[0]
                    event_id = event_id.split("_")[1]
                    events.setdefault(event_id, {})['hpe_detection'] = os.path.join(self.hpe_detection_dir, file_name)

        
        if self.hpe_sort == 'top_down':
            for file_name in os.listdir(self.hpe_detection_dir):
                if file_name.endswith('_pose_output.csv'):
                    event_name = os.path.splitext(file_name)[0] 
                    event_name = "_".join(event_name.split("_")[:2]) 
                    event_id = event_name.split("_")[1]
                    events.setdefault(event_id, {})['hpe_detection'] = os.path.join(self.hpe_detection_dir, file_name)

        for event_folder in os.listdir(self.images_base_dir):
            event_path = os.path.join(self.images_base_dir, event_folder)
            if os.path.isdir(event_path):
                event_id = event_folder
                event_id = event_id.split("_")[1]
                events.setdefault(event_id, {})['images_folder'] = event_path

        for event_folder in os.listdir(self.keypoints_field_dir):
            subfolder_path = os.path.join(self.keypoints_field_dir, event_folder)

            if os.path.isdir(subfolder_path):  # Check if it is a subfolder
                event_id = event_folder # Use the subfolder name as event_id
                event_id = event_id.split("_")[1]
                file_name = os.path.join(subfolder_path, 'predictions.pth')
                events.setdefault(event_id, {})['keypoints_field'] = os.path.join(self.keypoints_field_dir, file_name)
        return events
    

    #These following functions are all used to retrieve that certain data path for the given event_id. 
    def get_ball_detection(self, event_id):
        return self.events.get(event_id, {}).get('ball_detection')

    def get_players_detection(self, event_id):
        return self.events.get(event_id, {}).get('players_detection')

    def get_HPE(self, event_id):
        return self.events.get(event_id, {}).get('hpe_detection')

    def get_images_folder(self, event_id):
        return self.events.get(event_id, {}).get('images_folder')

    def get_keypoints_field(self, event_id):
        return self.events.get(event_id, {}).get('keypoints_field')