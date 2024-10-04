import sys
sys.path.append('/home/ivanmiert/sportlight_folder/soccernet-calibration-sportlight/')
import cv2
import os
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import concurrent.futures
import traceback
import json

def load_subfolders_from_json(json_file_path):
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            # Extract the subfolder values from each item in the list
            subfolders = [entry['subfolder'] for entry in data if 'subfolder' in entry]
            return subfolders
    except Exception as e:
        print(f"Error loading or parsing JSON file: {e}")
        return []
    
def transform_image_tensor(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor

def batch_images_to_tensors(base_image_folder):
    image_files = sorted(os.listdir(base_image_folder))
    selected_files = image_files
    print(f"selected files:{selected_files}")
    tensor_list = []
    for image_file in selected_files:
        image_path = os.path.join(base_image_folder, image_file)
        tensor = transform_image_tensor(image_path)
        tensor_list.append(tensor)
    return tensor_list

def get_predictions(tensor_list, model):
    if len(tensor_list) == 0:
        print("No tensors in the list to process.")
        return
    predictions = []
    batch_size = 8
    with torch.no_grad():
        for i in range(0, len(tensor_list), batch_size):
            # Ensure batch_tensors are properly shaped for the model
            batch_tensors = torch.cat(tensor_list[i:i + batch_size], dim=0)  # Ensure tensors are concatenated along the batch dimension
            # # Check if DataParallel is used and reshape accordingly
            # if isinstance(model, torch.nn.DataParallel): #if isinstance(model.nn_module, torch.nn.DataParallel):
            #     # Remove any additional dimension that may cause shape issues
            #     batch_tensors = batch_tensors.squeeze(1)
            # Remove any additional dimension that may cause shape issues
            batch_tensors = batch_tensors.squeeze(1)
            batch_predictions = model.predict(batch_tensors)
            for prediction in batch_predictions:
                    predictions.append(prediction.cpu().numpy())
                    #predictions.extend(prediction.cpu().numpy())
            #predictions.extend(batch_predictions.cpu().numpy())
    return predictions
    
def process_subfolder(clip_folder, model, image_base_folder, output_base_folder):
    clip_folder_path = os.path.join(image_base_folder, clip_folder)
    prediction_output_folder = os.path.join(output_base_folder, clip_folder)

    if os.path.exists(prediction_output_folder) and os.path.exists(os.path.join(prediction_output_folder, 'predictions.pth')):
            print(f"Predictions for {clip_folder} already exist, skipping.")
            return
    
    if os.path.isdir(clip_folder_path):
        print('Processing folder:', clip_folder_path)
        tensor_list = batch_images_to_tensors(clip_folder_path)
        if not tensor_list:
                print("Tensor list is empty, skipping.")
                return
        
        predictions = get_predictions(tensor_list, model)
        os.makedirs(prediction_output_folder, exist_ok=True)
        output_file = os.path.join(prediction_output_folder, 'predictions.pth')
        torch.save(predictions, output_file)
        print('Predictions have been saved')
        #del tensor_list
        #del predictions
        #torch.cuda.empty_cache()
        print(f"Predictions for {clip_folder} saved to {output_file}")


    
def perform_on_subfolders(image_base_folder, model, output_base_folder):
    #print(f"Performing on subfolders in base folder: {image_base_folder}")
    clip_folders = sorted(os.listdir(image_base_folder))
    #print(f"Clip folders found: {clip_folders}")

    # Filter clip_folders to include only those in subfolders_to_use
    #clip_folders = [folder for folder in clip_folders if folder in subfolders_to_use]
    #print(f"Filtered clip folders: {clip_folders}")

    for clip_folder in clip_folders:
        process_subfolder(clip_folder, model, image_base_folder, output_base_folder)


if __name__ == '__main__':
    image_base_folder = '/scratch-shared/ivanmiert/selected_frames_for_tests'
    output_base_folder = '/scratch-shared/ivanmiert/test_set/calibration2'
    model_path = '/scratch-shared/ivanmiert/data/model_finals_full.pt'
    #json_file_path = '/scratch-shared/ivanmiert/processed_subfolders.json'
    #subfolders_to_use = load_subfolders_from_json(json_file_path)
    print('starten')
    model = torch.load(model_path)
    model.eval()

    perform_on_subfolders(image_base_folder, model, output_base_folder)