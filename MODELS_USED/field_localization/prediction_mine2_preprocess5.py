import sys
import os
import torch
import cv2
import random
import json

sys.path.append('/home/ivanmiert/sportlight_folder/soccernet-calibration-sportlight/')
from metamodel import HRNetMetaModel

def prepare_tensor(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Normalize to [0, 1]
    return tensor

def extract_keypoints(output, confidence_threshold=0.5, min_keypoints=4):
    keypoints_with_confidence = []

    for i, keypoint in enumerate(output[0]):  # Assuming output shape is [1, num_keypoints, 3]
        x, y, confidence = keypoint[0].item(), keypoint[1].item(), keypoint[2].item()
        if confidence > confidence_threshold:
            keypoints_with_confidence.append((i, x, y, confidence))

    # Sort keypoints by confidence in descending order
    keypoints_with_confidence.sort(key=lambda x: x[3], reverse=True)

    # Check if there are at least `min_keypoints` keypoints with confidence > `confidence_threshold`
    if len(keypoints_with_confidence) >= min_keypoints:
        return True
    else:
        return False

def process_subfolder(subfolder_path, model, confidence_threshold=0.5, min_keypoints=4):
    frame_files = sorted([os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith('.jpg')])

    if len(frame_files) < 5:
        return None

    selected_frames = [
        frame_files[0],
        frame_files[len(frame_files) // 4],
        frame_files[len(frame_files) // 2],
        frame_files[3 * len(frame_files) // 4],
        frame_files[-1]
    ]

    all_frames_valid = True

    for frame_file in selected_frames:
        input_tensor = prepare_tensor(frame_file)

        with torch.no_grad():
            output = model.predict(input_tensor)
            #print(output)

        is_valid = extract_keypoints(output, confidence_threshold, min_keypoints)
        #print(is_valid)
        if not is_valid:  # If any frame is invalid, mark the subfolder as invalid
            all_frames_valid = False
            break

    if all_frames_valid:
        return {'subfolder': os.path.basename(subfolder_path), 'frames': selected_frames}
    else:
        print('not valid')
        return None

def process_random_subfolders(base_folder, model, num_subfolders=3000, confidence_threshold=0.5):
    all_subfolders = [os.path.join(base_folder, d) for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    #print('hier al subfolders:')
    #print(all_subfolders)
    random.shuffle(all_subfolders)

    processed_subfolders = []
    valid_count = 0
    i = 0
    #print(i)
    for subfolder in all_subfolders:
        #print(subfolder)
        #print(i)
        i+=1
        if valid_count >= num_subfolders:
            break

        keypoints_data = process_subfolder(subfolder, model, confidence_threshold)
        if keypoints_data:
            processed_subfolders.append({
                'subfolder': os.path.basename(subfolder),
                'frames': keypoints_data
            })
            valid_count += 1
            print(f"Processed {valid_count}/{num_subfolders} subfolders...")

    return processed_subfolders

# Load model
checkpoint_path = '/scratch-shared/ivanmiert/data/model_finals_full.pt'
model = torch.load(checkpoint_path)
model.eval()

# Base folder containing all event subfolders
base_folder_path = '/scratch-shared/ivanmiert/final_events_folder/frames_10000'
print('we doen iets:')
# Process subfolders and save results
processed_data = process_random_subfolders(base_folder_path, model, num_subfolders=3000)

# Save the list of processed subfolders and their corresponding keypoints
output_file_path = '/scratch-shared/ivanmiert/processed_subfolders.json'
with open(output_file_path, 'w') as f:
    json.dump(processed_data, f, indent=4)

print(f"Processing complete. Data saved to {output_file_path}.")