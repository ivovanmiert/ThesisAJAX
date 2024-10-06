import cv2
import os
import yaml

""" This file converts the sliced clips that are available in the scratch-shared environment to seperate frames. 
    It stores the frames for each event/sliced clip in a seperate folder, these can later be used for further processing.  
"""


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config('/home/ivanmiert/frames_home/config.yaml')

clips_folder = config['data_paths']['sliced_clips_base_folder']
output_folder = config['data_paths']['frames_sliced_clips_base_folder']

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over each file in the clips folder
for clip_name in os.listdir(clips_folder):
    clip_path = os.path.join(clips_folder, clip_name)
    
    # Create a directory for this clip's frames
    clip_output_folder = os.path.join(output_folder, os.path.splitext(clip_name)[0])
    if not os.path.exists(clip_output_folder):
        os.makedirs(clip_output_folder)
    
    # Capture the video
    video = cv2.VideoCapture(clip_path)
    
    frame_number = 0
    while True:
        success, frame = video.read()
        if not success:
            break
        
        # Save the frame as an image file
        frame_filename = os.path.join(clip_output_folder, f'frame_{frame_number:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        
        frame_number += 1

    # Release the video capture object
    video.release()

print("Frame extraction complete.")
