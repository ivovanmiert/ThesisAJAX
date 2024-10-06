import os 
import cv2

"""
This class is regarding to make it possible to easily access the frames of a certain event. 
This is done at the start of processing an event, so from this object, the different models can easily access the frames to process. 
"""


class FramesObject:
    def __init__(self, clip_folder_path):
        """
        Initializes the FramesObject with the path to the clip folder containing frames
        :param clip_folder_path: Path to the folder containing the frames for the clip
        """
        self.clip_folder_path = clip_folder_path
        self.frames = self.load_frames(clip_folder_path)
    
    def load_frames(self, clip_folder_path):
        """
        Loads all frames from the specified folder.
        :param clip_folder_path: Path to the folder containing frames
        :return: Dictionary mapping frame file names to frame images
        """
        frames = {}
        for frame_file in sorted(os.listdir(clip_folder_path)):
            if frame_file.endswith('.jpg') or frame_file.endswith('.png'):
                frame_path = os.path.join(clip_folder_path, frame_file)
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Extract the numeric part of the frame file name
                frame_index = int(frame_file.split('_')[1].split('.')[0])
                frames[frame_index] = frame
        return frames
    
    def get_frame(self, frame_id):
        """
        Returns the frame image for the specified frame file name
        :param frame_file: File name of the file
        :return: Frame image
        """
        return self.frames.get(frame_id)
    

    
