import sys
import os
import cv2
import torch
import traceback
import random
import numpy as np

sys.path.append('/home/ivanmiert/sportlight_folder/soccernet-calibration-sportlight/')

def transform_image_tensor(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor

def select_random_frames(image_files, num_frames=5):
    total_frames = len(image_files)
    step = max(total_frames // num_frames, 1)
    selected_indices = list(range(0, total_frames, step))[:num_frames]
    return [image_files[i] for i in selected_indices]

def batch_images_to_tensors(base_image_folder, num_frames=5):
    image_files = sorted(os.listdir(base_image_folder))
    selected_files = select_random_frames(image_files, num_frames=num_frames)
    print(f"Selected files: {selected_files}")
    tensor_list = []
    for image_file in selected_files:
        image_path = os.path.join(base_image_folder, image_file)
        tensor = transform_image_tensor(image_path)
        tensor_list.append(tensor)
    return tensor_list

def get_predictions(tensor_list, model):
    if len(tensor_list) == 0:
        print("No tensors in the list to process.")
        return []
    predictions = []
    model.eval()
    batch_size = 8
    with torch.no_grad():
        for i in range(0, len(tensor_list), batch_size):
            batch_tensors = torch.cat(tensor_list[i:i + batch_size], dim=0)
            batch_tensors = batch_tensors.to(next(model.parameters()).device)
            if isinstance(model, torch.nn.DataParallel):
                batch_tensors = batch_tensors.squeeze(1)
            batch_predictions = model(batch_tensors)
            for prediction in batch_predictions:
                predictions.append(prediction.cpu().numpy())
    return predictions

def evaluate_predictions(predictions):
    valid_frames = 0

    for frame_keypoints in predictions:
        keypoints_with_confidence = []

        # Skip indices similar to how it's done in the estimate_homography function
        skip_indices = {0, 1, 24, 25}
        for i, keypoint in enumerate(frame_keypoints[0]):
            print(f"keypoint: {keypoint}")
            if i in skip_indices:
                continue
            if keypoint is not None and len(keypoint) > 2:
                keypoints_with_confidence.append((i, keypoint[0].item(), keypoint[1].item(), keypoint[2].item()))

        # Sort keypoints based on confidence in descending order
        keypoints_with_confidence.sort(key=lambda x: x[3], reverse=True)

        # Select the top 4 keypoints based on confidence
        top_keypoints = keypoints_with_confidence[:4]
        print(top_keypoints)
        # Check if all top 4 keypoints have confidence > 0.5
        if len(top_keypoints) == 4 and all(kp[3] > 0.5 for kp in top_keypoints):
            valid_frames += 1

    # If all 5 frames are valid, return True, else return False
    return valid_frames == 5

def process_clip_folder(clip_folder, image_base_folder, model):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.nn_module.to(device)
        model = torch.nn.DataParallel(model.nn_module)

        clip_folder_path = os.path.join(image_base_folder, clip_folder)
        if os.path.isdir(clip_folder_path):
            tensor_list = batch_images_to_tensors(clip_folder_path)
            if not tensor_list:
                return False
            tensor_list = [tensor.to(device) for tensor in tensor_list]
            predictions = get_predictions(tensor_list, model)
            return evaluate_predictions(predictions)

    except Exception as e:
        print(f"An error occurred in folder {clip_folder}: {str(e)}")
        traceback.print_exc()
        return False

def perform_on_subfolders(image_base_folder, model, max_folders=3000):
    clip_folders = sorted(os.listdir(image_base_folder))
    random.shuffle(clip_folders)
    selected_folders = []
    i = 0
    for clip_folder in clip_folders:
        print(i)
        i += 1
        if len(selected_folders) >= max_folders:
            break
        if process_clip_folder(clip_folder, image_base_folder, model):
            selected_folders.append(clip_folder)
    
    return selected_folders

def save_selected_folders(selected_folders, output_file_path):
    try:
        with open(output_file_path, 'w') as file:
            for folder in selected_folders:
                file.write(f"{folder}\n")
        print(f"Selected folders saved to {output_file_path}")
    except Exception as e:
        print(f"An error occurred while saving the selected folders: {str(e)}")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    image_base_folder = '/scratch-shared/ivanmiert/final_events_folder/frames_10000'
    model_path = '/scratch-shared/ivanmiert/data/model_finals_full.pt'
    output_list = '/home/ivanmiert/sportlight_folder/soccernet-calibration-sportlight/src/models/hrnet/selected_folders_test.txt'


    try:
        model = torch.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
    model.eval()
    selected_folders = perform_on_subfolders(image_base_folder, model)
    print(f"Selected folders: {selected_folders}")
    print(f"Total selected folders: {len(selected_folders)}")

    save_selected_folders(selected_folders, output_list)