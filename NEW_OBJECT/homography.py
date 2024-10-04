import numpy as np
import cv2
import torch
import json

class HomographyEstimator:
    def __init__(self, field_keypoints, predictions_file):
        self.field_keypoints = field_keypoints.get_keypoints()  # Get the keypoints dictionary directly
        self.homographies = {}
        self.available_frames = []
        self.frame_interval=3
        self.valid_homographies = False 

        self.predictions = torch.load(predictions_file, map_location=torch.device('cpu'))

        for frame_idx, frame_keypoints in enumerate(self.predictions):
            actual_frame_idx = frame_idx * self.frame_interval
            self.available_frames.append(actual_frame_idx)
            self.estimate_homography(frame_keypoints, actual_frame_idx)
        if not self.valid_homographies:
            print("No valid homographies found. Skipping the clip.")
        else:
            self.homographies = self.interpolate_missing_homographies(self.homographies)
            self.interpolate_homographies()

    def estimate_homography(self, frame_keypoints, frame_index):
        image_points = []
        planar_points = []
        keypoints_with_confidence = []

        skip_indices = {0, 1, 24, 25}
        for i, keypoint in enumerate(frame_keypoints):
            if i in skip_indices:
                continue
            if keypoint is not None and len(keypoint) > 2:
                keypoints_with_confidence.append((i, keypoint[0].item(), keypoint[1].item(), keypoint[2].item(), self.field_keypoints[i]))

        keypoints_with_confidence.sort(key=lambda x: x[3], reverse=True)
        # Select keypoints with confidence greater than 0.5
        top_keypoints = [kp for kp in keypoints_with_confidence if kp[3] > 0.5]

        for kp in top_keypoints:
            image_points.append((kp[1]*(1280/960), kp[2]*(720/540)))
            planar_points.append((kp[4][0], kp[4][1]))

        if len(image_points) >= 4:
            self.valid_homographies = True
            image_points = np.array(image_points, dtype='float32')
            planar_points = np.array(planar_points, dtype='float32')
            H, mask = cv2.findHomography(image_points, planar_points, cv2.RANSAC, 1.0)
            self.homographies[frame_index] = H
        else:
            print('no homography available:')
            self.homographies[frame_index] = None

    def interpolate_missing_homographies(self, homographies):
        #This function is meant to interpolate the 3rd frames homographies that are missing. 
        available_frames = sorted([frame for frame, H in homographies.items() if H is not None])
        
        # Interpolation for missing homographies
        all_frames = sorted(homographies.keys())
        for frame in all_frames:
            if homographies[frame] is None:
                # Find the closest previous and next available frames with homographies
                prev_frame = max([f for f in available_frames if f < frame], default=None)
                next_frame = min([f for f in available_frames if f > frame], default=None)
                
                if prev_frame is not None and next_frame is not None:
                    # Interpolate between the previous and next homographies
                    alpha = (frame - prev_frame) / (next_frame - prev_frame)
                    homographies[frame] = (1 - alpha) * homographies[prev_frame] + alpha * homographies[next_frame]
                elif prev_frame is None and next_frame is not None:
                    # If there's no previous homography, use the next available one
                    homographies[frame] = homographies[next_frame]
                elif next_frame is None and prev_frame is not None:
                    # If there's no next homography, use the previous available one
                    homographies[frame] = homographies[prev_frame]
        
        return homographies
    
    def interpolate_homographies(self):
        all_frames = list(range(min(self.available_frames), max(self.available_frames) + 2))
        for i in range(len(self.available_frames) - 1):
            start_frame = self.available_frames[i]
            end_frame = self.available_frames[i + 1]
            H_start = self.homographies[start_frame]
            H_end = self.homographies[end_frame]

            if H_start is not None and H_end is not None:
                for j in range(1, self.frame_interval):
                    interp_frame = start_frame + j
                    alpha = j / self.frame_interval
                    # Linear interpolation of homographies
                    H_interp = (1 - alpha) * H_start + alpha * H_end
                    self.homographies[interp_frame] = H_interp

        # Propagate the last available homography to the subsequent frames
        last_available_frame = self.available_frames[-1]
        last_homography = self.homographies[last_available_frame]
        for frame in range(last_available_frame + 1, last_available_frame + 3):

            self.homographies[frame] = last_homography
        
    def get_homography(self, frame_index):
        if 0 <= frame_index < len(self.homographies):
            return self.homographies[frame_index]
        else:
            #print(self.homographies)
            raise IndexError("Frame Index out of range")

    def warp_points(self, frame_index, points, player_id):
        image_points_homogeneous = np.array([points[0], points[1], 1.0])
        H = self.get_homography(frame_index)

        if H is not None:
            points = np.array(points, dtype='float32').reshape(-1, 1, 2)
            warped_points = cv2.perspectiveTransform(points, H)
            return warped_points
        else:
            print("Homography matrix is not available for frame index:", frame_index)
            return None