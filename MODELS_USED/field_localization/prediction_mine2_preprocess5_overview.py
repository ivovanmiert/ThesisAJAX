import sys
import os
import torch
import cv2
import random
import json

sys.path.append('/home/ivanmiert/sportlight_folder/soccernet-calibration-sportlight/')
from metamodel import HRNetMetaModel

class ImageProcessor:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = torch.load(model_path)
        model.eval()
        return model

    def prepare_tensor(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Normalize to [0, 1]
        return tensor

    def extract_keypoints(self, output, confidence_threshold=0.5, min_keypoints=4):
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

    def process_subfolder(self, subfolder_path, confidence_threshold=0.5, min_keypoints=4):
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
            input_tensor = self.prepare_tensor(frame_file)

            with torch.no_grad():
                output = self.model.predict(input_tensor)

            is_valid = self.extract_keypoints(output, confidence_threshold, min_keypoints)
            if not is_valid:  # If any frame is invalid, mark the subfolder as invalid
                all_frames_valid = False
                break

        if all_frames_valid:
            return {'subfolder': os.path.basename(subfolder_path), 'frames': selected_frames}
        else:
            return None

    def process_random_subfolders(self, base_folder, num_subfolders=3000, confidence_threshold=0.5):
        all_subfolders = [os.path.join(base_folder, d) for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
        random.shuffle(all_subfolders)

        processed_subfolders = []
        valid_count = 0

        for subfolder in all_subfolders:
            if valid_count >= num_subfolders:
                break

            keypoints_data = self.process_subfolder(subfolder, confidence_threshold)
            if keypoints_data:
                processed_subfolders.append({
                    'subfolder': os.path.basename(subfolder),
                    'frames': keypoints_data
                })
                valid_count += 1
                print(f"Processed {valid_count}/{num_subfolders} subfolders...")

        return processed_subfolders


def main(model_path, image_base_folder, output_file_path, num_subfolders=1000):
    processor = ImageProcessor(model_path)
    processed_data = processor.process_random_subfolders(image_base_folder, num_subfolders=num_subfolders)

    with open(output_file_path, 'w') as f:
        json.dump(processed_data, f, indent=4)

    print(f"Processing complete. Data saved to {output_file_path}.")


# # The main function can be accessed externally like this:
# if __name__ == '__main__':
#     model_path = '/scratch-shared/ivanmiert/data/model_finals_full.pt'
#     image_base_folder = '/scratch-shared/ivanmiert/final_events_folder/frames_10000'
#     output_file_path = '/scratch-shared/ivanmiert/processed_subfolders.json'
    
#     main(model_path, image_base_folder, output_file_path)