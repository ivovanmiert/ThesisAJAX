import os
import re



"""This file was used to find the intersection of the events processed by the different models. 
Only if all models had processed that event, the event can be processed to obtain features.  
It was done with either "pifpaf" or "top_down" based on which of the two HPE was handled. 
"""

def intersection(chunk):
    # Define the folder paths
    folders = {
        "ball_detection_processed": f"/scratch-shared/ivanmiert/overview/ball_detect_processed/chunk_{chunk}",
        "calibration": f"/scratch-shared/ivanmiert/overview/keypoints_pred/chunk_{chunk}",
        #"pifpaf": f"/scratch-shared/ivanmiert/overview/pifpaf_pred/chunk_{chunk}",
        "player_detections": f"/scratch-shared/ivanmiert/overview/player_detect/chunk_{chunk}",
        "top_down": f"/scratch-shared/ivanmiert/overview/top_down_pred/chunk_{chunk}"
    }

    # Extract event IDs from files/folders
    def extract_event_ids(folder_path, pattern):
        event_ids = set()
        for item in os.listdir(folder_path):
            match = re.search(pattern, item)
            if match:
                event_ids.add(match.group(1))
        return event_ids

    # Define the patterns to extract event IDs from filenames/folder names
    patterns = {
        "ball_detection_processed": r"event_(\d+)\.csv",
        "calibration": r"event_(\d+)$",
        #"pifpaf": r"event_(\d+)\.csv",
        "player_detections": r"event_(\d+)$",
        "top_down": r"event_(\d+)_pose_output\.csv"
    }

    # Extract event IDs from each folder
    event_ids = {}
    for folder_name, folder_path in folders.items():
        event_ids[folder_name] = extract_event_ids(folder_path, patterns[folder_name])

    #Intersection of all event IDs
    common_event_ids = set.intersection(*event_ids.values())

    # Convert the common event IDs to a sorted list
    common_event_ids_list = sorted(list(map(int, common_event_ids)))
    return common_event_ids_list