import sys
sys.path.append('/home/ivanmiert/sportlight_folder/soccernet-calibration-sportlight/')
import os
import cv2
import torch
import json

class FolderProcessor:
    def __init__(self, model_path, image_base_folder, output_base_folder):
        self.model = self.load_model(model_path)
        self.image_base_folder = image_base_folder
        self.output_base_folder = output_base_folder

    def load_model(self, model_path):
        model = torch.load(model_path)
        model.eval()
        return model

    @staticmethod
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

    @staticmethod
    def transform_image_tensor(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return tensor

    def batch_images_to_tensors(self, base_image_folder):
        image_files = sorted(os.listdir(base_image_folder))
        selected_files = image_files[::3]
        print(f"Selected files: {selected_files}")
        tensor_list = []
        for image_file in selected_files:
            image_path = os.path.join(base_image_folder, image_file)
            tensor = self.transform_image_tensor(image_path)
            tensor_list.append(tensor)
        return tensor_list

    def get_predictions(self, tensor_list):
        if len(tensor_list) == 0:
            print("No tensors in the list to process.")
            return []
        predictions = []
        batch_size = 8
        with torch.no_grad():
            for i in range(0, len(tensor_list), batch_size):
                # Ensure batch_tensors are properly shaped for the model
                batch_tensors = torch.cat(tensor_list[i:i + batch_size], dim=0)  # Concatenate along batch dimension
                batch_tensors = batch_tensors.squeeze(1)  # Remove extra dimension if needed
                batch_predictions = self.model.predict(batch_tensors)
                predictions.extend([prediction.cpu().numpy() for prediction in batch_predictions])
        return predictions

    def process_subfolder(self, clip_folder):
        clip_folder_path = os.path.join(self.image_base_folder, clip_folder)
        prediction_output_folder = os.path.join(self.output_base_folder, clip_folder)

        if os.path.exists(prediction_output_folder) and os.path.exists(os.path.join(prediction_output_folder, 'predictions.pth')):
            print(f"Predictions for {clip_folder} already exist, skipping.")
            return

        if os.path.isdir(clip_folder_path):
            print('Processing folder:', clip_folder_path)
            tensor_list = self.batch_images_to_tensors(clip_folder_path)
            if not tensor_list:
                print("Tensor list is empty, skipping.")
                return

            predictions = self.get_predictions(tensor_list)
            os.makedirs(prediction_output_folder, exist_ok=True)
            output_file = os.path.join(prediction_output_folder, 'predictions.pth')
            torch.save(predictions, output_file)
            print(f"Predictions for {clip_folder} saved to {output_file}")

    def perform_on_subfolders(self, subfolders_to_use):
        clip_folders = sorted(os.listdir(self.image_base_folder))
        clip_folders = [folder for folder in clip_folders if folder in subfolders_to_use]

        for clip_folder in clip_folders:
            self.process_subfolder(clip_folder)


def main(model_path, image_base_folder, output_base_folder, json_file_path):
    processor = FolderProcessor(model_path, image_base_folder, output_base_folder)
    subfolders_to_use = processor.load_subfolders_from_json(json_file_path)
    if not subfolders_to_use:
        print("No subfolders to process.")
        return
    processor.perform_on_subfolders(subfolders_to_use)
    print("Processing completed.")




# The main function can be accessed externally like this:
if __name__ == '__main__':
    model_path = '/scratch-shared/ivanmiert/data/model_finals_full.pt'
    image_base_folder = '/scratch-shared/ivanmiert/final_events_folder/frames_10000'
    output_base_folder = '/scratch-shared/ivanmiert/final_events_folder/calibration10000'
    json_file_path = '/scratch-shared/ivanmiert/processed_subfolders.json'

    print("Starting the process...")
    main(model_path, image_base_folder, output_base_folder, json_file_path)