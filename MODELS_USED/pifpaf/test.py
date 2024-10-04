import os

video_path = '/scratch-shared/ivanmiert/Sliced_Clips/1767073123.mp4'
if os.path.exists(video_path):
    print("Video file exists.")
else:
    print("Video file does not exist.")

import cv2

video_path = '/scratch-shared/ivanmiert/Sliced_Clips/1767073123.mp4'
capture = cv2.VideoCapture(video_path)

if not capture.isOpened():
    print("Error: Could not open video.")
else:
    print("Video opened successfully.")
    ret, frame = capture.read()
    if ret:
        print("Read a frame successfully.")
    else:
        print("Error: Could not read a frame.")

capture.release()