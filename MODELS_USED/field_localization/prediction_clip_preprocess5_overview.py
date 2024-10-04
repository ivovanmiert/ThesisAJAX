import sys
import os
import cv2
import torch
import traceback
import random
import numpy as np

sys.path.append('/home/ivanmiert/sportlight_folder/soccernet-calibration-sportlight/')

class ImageProcessor:
    def __init__(self, model_path, image_base_folder, output_file_path, num_frames=5, max_folders=3000):
        self.model_path = model_path
        self.image_base_folder = image_base_folder
        self.output_file_path = output_file_path
        self.num_frames = num_frames
        self.max_folders = max_folders
        self.model = self.load_model()

    def load_model(self):
        try:
            model = torch.load(self.model_path)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def transform_image_tensor(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return tensor

    def select_random_frames(self, image_files):
        total_frames = len(image_files)
        step = max(total_frames // self.num_frames, 1)
        selected_indices = list(range(0, total_frames, step))[:self.num_frames]
        return [image_files[i] for i in selected_indices]

    def batch_images_to_tensors(self, base_image_folder):
        image_files = sorted(os.listdir(base_image_folder))
        selected_files = self.select_random_frames(image_files)
        print(f"Selected files: {selected_files}")
        tensor_list = []
        for image_file in selected_files:
            image_path = os.path.join(base_image_folder, image_file)
            tensor = self.transform_image_tensor(image_path)
            tensor_list.append(tensor)
        return tensor_list

    def get_predictions(self, tensor_list, model):
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

    def evaluate_predictions(self, predictions):
        valid_frames = 0
        for frame_keypoints in predictions:
            keypoints_with_confidence = []
            skip_indices = {0, 1, 24, 25}
            for i, keypoint in enumerate(frame_keypoints[0]):
                if i in skip_indices:
                    continue
                if keypoint is not None and len(keypoint) > 2:
                    print(f"keypoint: {keypoint}")
                    keypoints_with_confidence.append((i, keypoint[0].item(), keypoint[1].item(), keypoint[2].item()))

            keypoints_with_confidence.sort(key=lambda x: x[3], reverse=True)
            top_keypoints = keypoints_with_confidence[:4]
            if len(top_keypoints) == 4 and all(kp[3] > 0.5 for kp in top_keypoints):
                valid_frames += 1

        return valid_frames == self.num_frames

    def process_clip_folder(self, clip_folder):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.nn_module.to(device)
            model = torch.nn.DataParallel(self.model.nn_module)

            clip_folder_path = os.path.join(self.image_base_folder, clip_folder)
            if os.path.isdir(clip_folder_path):
                tensor_list = self.batch_images_to_tensors(clip_folder_path)
                if not tensor_list:
                    return False
                tensor_list = [tensor.to(device) for tensor in tensor_list]
                predictions = self.get_predictions(tensor_list, model)
                return self.evaluate_predictions(predictions)

        except Exception as e:
            print(f"An error occurred in folder {clip_folder}: {str(e)}")
            traceback.print_exc()
            return False

    def perform_on_subfolders(self):
        clip_folders = sorted(os.listdir(self.image_base_folder))
        random.shuffle(clip_folders)
        selected_folders = []
        i = 0
        for clip_folder in clip_folders:
            print(i)
            i += 1
            if len(selected_folders) >= self.max_folders:
                break
            if self.process_clip_folder(clip_folder):
                selected_folders.append(clip_folder)

        return selected_folders

    def save_selected_folders(self, selected_folders):
        try:
            with open(self.output_file_path, 'w') as file:
                for folder in selected_folders:
                    file.write(f"{folder}\n")
            print(f"Selected folders saved to {self.output_file_path}")
        except Exception as e:
            print(f"An error occurred while saving the selected folders: {str(e)}")


# External usage
def main(model_path, image_base_folder, output_file_path):
    processor = ImageProcessor(model_path, image_base_folder, output_file_path)
    selected_folders = processor.perform_on_subfolders()
    print(f"Selected folders: {selected_folders}")
    print(f"Total selected folders: {len(selected_folders)}")
    processor.save_selected_folders(selected_folders)


# If executed directly, will call the main function
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    
    image_base_folder = '/scratch-shared/ivanmiert/final_events_folder/frames_10000'
    model_path = '/scratch-shared/ivanmiert/data/model_finals_full.pt'
    output_list = '/home/ivanmiert/sportlight_folder/soccernet-calibration-sportlight/src/models/hrnet/selected_folders_test.txt'

    main(model_path, image_base_folder, output_list)