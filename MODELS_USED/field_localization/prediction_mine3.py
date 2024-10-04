import sys
sys.path.append('/home/ivanmiert/sportlight_folder/soccernet-calibration-sportlight/')
from metamodel import HRNetMetaModel
import torch
import cv2

import os
import cv2
import torch

def prepare_tensor(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Normalize to [0, 1]
    return tensor

def process_folder(model, input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Process image files
            image_path = os.path.join(input_folder, filename)
            input_tensor = prepare_tensor(image_path)
            
            # Perform inference
            with torch.no_grad():
                output = model.predict(input_tensor)
            
            # Save the output tensor
            output_filename = os.path.splitext(filename)[0] + '_output.pt'
            output_path = os.path.join(output_folder, output_filename)
            torch.save(output, output_path)

# Example usage
checkpoint_path = '/scratch-shared/ivanmiert/data/model_finals_full.pt'
input_folder = '/scratch-shared/ivanmiert/juv_bol/1/'
output_folder = '/scratch-shared/ivanmiert/juv_bol/tensor_outputs/'

# Load the model
model = torch.load(checkpoint_path)
model.eval()

# Process the folder
process_folder(model, input_folder, output_folder)