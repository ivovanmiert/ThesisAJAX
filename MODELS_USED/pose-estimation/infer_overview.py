import torch
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms as T
from PIL import Image
import os
import time

from pose.models import get_pose_model
from pose.utils.boxes import letterbox, scale_boxes, non_max_suppression, xyxy2xywh
from pose.utils.decode import get_final_preds, get_simdr_final_preds
from pose.utils.utils import setup_cudnn, get_affine_transform, draw_keypoints
from pose.utils.utils import VideoReader, VideoWriter, WebcamStream, FPS
import pandas as pd

import sys
sys.path.insert(0, '/home/ivanmiert/pose-estimation/yolov5')
#sys.path.insert(0, '/home/ivanmiert/pose-estimation/yolov5/models')
from yolov5.models.experimental import attempt_load


class Pose:
    def __init__(self, 
        #det_model,
        pose_model,
        img_size=640,
        conf_thres=0.25,
        iou_thres=0.45, 
    ) -> None:
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.det_model = attempt_load(det_model, map_location=self.device)
        #self.det_model = self.det_model.to(self.device)

        self.model_name = pose_model
        self.pose_model = get_pose_model(pose_model)
        self.pose_model.load_state_dict(torch.load(pose_model, map_location=self.device)) #map_location='cpu'))
        self.pose_model = self.pose_model.to(self.device)
        self.pose_model.eval()

        self.patch_size = (192, 256)

        self.pose_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.coco_skeletons = [
            [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
            [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
        ]

    def preprocess(self, image):
        img = letterbox(image, new_shape=self.img_size)
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        img = img[None]
        return img

    def box_to_center_scale(self, boxes, pixel_std=200):
        #print('dit is hoe ze erin gaan:')
        #print(boxes)
        #boxes = xyxy2xywh(boxes)
        #print('dit is hoe ze eruit komen:')
        #print(boxes)
        r = self.patch_size[0] / self.patch_size[1]
        #print('Hier de patch_size: Dit is dus de size van de patch over welke de heatmap van 48 x 64 gemaakt wordt. Er moet gekeken worden of deze patch size ook direct uit de afbeelding genomen wordt. Is deze ook voor iedere box hetzelfde? Dan moet de heatmap prediction worden omgezet naar die omvang van de patch en dan op de juiste plek in de afbeelding worden gezet.')
        #print(r) #0.75
        #print('Hier de afmetingen van patch size, als het goed is 192, 256')
        #print(self.patch_size[0])
        #print(self.patch_size[1])
        mask = boxes[:, 2] > boxes[:, 3] * r
        boxes[mask, 3] = boxes[mask, 2] / r
        boxes[~mask, 2] = boxes[~mask, 3] * r
        #boxes[:, 2:] /= pixel_std 
        #boxes[:, 2:] *= 1.25
        #print('hier de boxes na al die dingen:')
        #print(boxes)
        return boxes
    
        # Function to convert bounding box coordinates to (cx, cy, w, h)
    def convert_and_scale_bbox_format(self, bboxes, original_size, target_size):
        original_width, original_height = original_size
        target_width, target_height = target_size
        
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        
        converted_boxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            if x1 == -1 or y1 == -1 or x2 == -1 or y2 == -1:
                continue  # Skip invalid bounding boxes
            
            # Scale coordinates
            # x1 = int(x1 * scale_x)
            # y1 = int(y1 * scale_y)
            # x2 = int(x2 * scale_x)
            # y2 = int(y2 * scale_y)
            #print(x1, y1, x2, y2)
            
            # Convert to (cx, cy, w, h)
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            converted_boxes.append([cx, cy, w, h])
        #print(converted_boxes)
        return converted_boxes


    def predict_poses(self, boxes, img):
        image_patches = []
        #print(boxes)
        list_width_height = []
        #save_dir = "/scratch-shared/ivanmiert/wasb"
        for idx, (cx, cy, w, h) in enumerate(boxes):
            #print('loop')
            #print(cx, cy, w, h)
            trans = get_affine_transform(np.array([cx, cy]), np.array([w, h]), self.patch_size)
            #print(trans)
            img_patch = cv2.warpAffine(img, trans, self.patch_size, flags=cv2.INTER_LINEAR)
            #print('eerste img_patch HIER:')
            #print(img_patch)
            img_patch = self.pose_transform(img_patch)
            #print('tweede img patch HIER:')
            #print(img_patch)
            image_patches.append(img_patch)
            list_width_height.append((w,h))

            # Save the transformed image patch as a PNG file
            img_patch_numpy = img_patch.permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array
            img_patch_numpy = (img_patch_numpy * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
            #img_patch_pil = Image.fromarray(img_patch_numpy)
            #img_patch_pil.save(os.path.join(save_dir, f'patch_{idx}.png'))

        #print('Hier list_width_height:')
        #print(list_width_height)
        image_patches = torch.stack(image_patches).to(self.device)
        #print('image_patches HIER:')
        #print(image_patches)
        #print('shape van image_patches hier:')
        #print(image_patches.shape)
        return self.pose_model(image_patches), list_width_height

    def postprocess(self, det, img1, img0):
        #pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=0)

        #print(det)
        #print(img0.shape[:2])
        #print(img1.shape[-2])
        # if len(det):
        #     boxes = scale_boxes(det, img0.shape[:2], img1.shape[-2:]).cpu()
        #     boxes = self.box_to_center_scale(boxes)
        #     outputs = self.predict_poses(boxes, img0)

        #     if 'simdr' in self.model_name:
        #         coords = get_simdr_final_preds(*outputs, boxes, self.patch_size)
        #     else:
        #         coords = get_final_preds(outputs, boxes)

        #     draw_keypoints(img0, coords, self.coco_skeletons)

        if len(det):
            # Convert det to a NumPy array if it's a PyTorch tensor
            if isinstance(det, torch.Tensor):
                det = det.numpy()

            #print('img 0 shape:')
            #print(img0.shape[:2])
            #print('img1 shape:')
            #print(img1.shape[-2:])
            #print('Deze shapes worden dus gebruikt in de functie scale_boxes')
            boxes = scale_boxes(det, img0.shape[:2], img1.shape[-2:])
            
            # Convert boxes back to a PyTorch tensor
            boxes = torch.from_numpy(boxes).float().cpu()

            boxes = self.box_to_center_scale(boxes)
            #print('hier de boxes: Er moet gekeken worden of deze boxes dezelfde dingen bevat die nodig zijn in de predict_poses functie, als in de get_final_preds functie.')
            #print(boxes)
            outputs, list_width_height = self.predict_poses(boxes, img0)
            #print('Hier de outputs van de forward function, al gereturned van de andere functie:')
            #print(outputs)
            #print('En nu de shapes van die outputs:')
            #print(outputs.shape)

            if 'simdr' in self.model_name:
                coords = get_simdr_final_preds(*outputs, boxes, self.patch_size)
            else:
                coords = get_final_preds(outputs, boxes, list_width_height)
            #print('hier de coords coords:')
            #print(coords)
            return coords
            #save_path = '/scratch-shared/ivanmiert/output_posenet_2.png'
            #draw_keypoints(img0, coords, self.coco_skeletons, save_path)

    @torch.no_grad()
    def predict(self, image, pred):
        img = self.preprocess(image)
        #pred = self.det_model(img)[0]  
        coords = self.postprocess(pred, img, image)
        return coords


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/scratch-shared/ivanmiert/pose_frame_00000.png')
    #parser.add_argument('--det-model', type=str, default='')
    parser.add_argument('--pose-model', type=str, default='/home/ivanmiert/pose-estimation/posehrnet_w48_256x192.pth')#pose_hrnet_w48_256x192.pth')
    parser.add_argument('--img-size', type=int, default=1280)
    parser.add_argument('--conf-thres', type=float, default=0.4)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    return parser.parse_args()



class PosePredictionPipeline:
    def __init__(self, pose_model, img_size, conf_thres, iou_thres):
        self.pose = Pose(
            pose_model,
            img_size,
            conf_thres,
            iou_thres
        )

    def process_event(self, detections_folder, frames_folder, output_folder, original_size, target_size, subfolders_to_process):
        """
        Process each event in the detections folder, overlay poses on frames, and save the output.
        Parameters:
            detections_folder (str): Path to the folder containing detection CSV files.
            frames_folder (str): Path to the folder containing frames for events.
            output_folder (str): Path where the pose predictions will be saved.
            original_size (tuple): Original size of the frames.
            target_size (tuple): Target size for scaling bounding boxes.
            event_range (tuple): Tuple containing the min and max event_id to process (optional).
        """
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Iterate over all the events in the detection folder
        for event_dir in os.listdir(detections_folder):
            event_id = event_dir.split('_')[-1]  # Extract the event_id from the folder name

            if event_id not in subfolders_to_process:
                print('skippa')
                continue

            output_csv_path = os.path.join(output_folder, f'event_{event_id}_pose_output.csv')

            # Skip if the output CSV already exists
            if os.path.exists(output_csv_path):
                print(f'File for event {event_id} already exists. Skipping...')
                continue

            csv_file_path = os.path.join(detections_folder, event_dir, 'out.csv')

            # Check if the CSV file exists
            if not os.path.exists(csv_file_path):
                print(f"CSV file not found: {csv_file_path}")
                continue

            # Start processing event
            self.process_csv(csv_file_path, frames_folder, event_id, output_csv_path, original_size, target_size)

    def process_csv(self, csv_file_path, frames_folder, event_id, output_csv_path, original_size, target_size):
        """
        Processes a single CSV file with detection data.
        """
        start_time = time.time()

        # Read the detection CSV file
        df = pd.read_csv(csv_file_path, header=None)

        # Extract the relevant columns: frame_number, detection_id, and bbox coordinates
        df = df.iloc[:, 0:6]  # Assuming structure: frame_number, detection_id, xmin, ymin, xmax, ymax

        # Dictionary to store the output for each frame
        output_data = []

        # Group data by frame number
        df = df.groupby(df.columns[0])

        # Iterate through each group (each frame)
        for frame_number, group in df:
            frame_number = int(frame_number)
            frame_path = os.path.join(frames_folder, f'event_{event_id}', f'frame_{frame_number:04d}.jpg')

            # Check if the frame image exists
            if not os.path.exists(frame_path):
                print(f"Frame not found: {frame_path}")
                continue

            # Load the image once per frame
            image = cv2.imread(frame_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Prepare bounding boxes for the current frame
            all_bbox_coords, detection_ids = self.prepare_bboxes(group)

            if not all_bbox_coords:
                continue

            # Convert and scale bounding boxes
            scaled_bboxes = self.pose.convert_and_scale_bbox_format(all_bbox_coords, original_size, target_size)

            # Predict poses for all bounding boxes
            pose_predictions = self.pose.predict(image, scaled_bboxes)

            # Append output data
            for detection_id, bbox_coords, prediction in zip(detection_ids, all_bbox_coords, pose_predictions):
                flattened_predictions = prediction.flatten().tolist()
                output_data.append([frame_number, detection_id] + bbox_coords + flattened_predictions)

        # Save output to CSV
        self.save_output(output_data, output_csv_path)

        end_time = time.time()
        print(f"Total time for event {event_id}: {end_time - start_time} seconds")

    def prepare_bboxes(self, group):
        """
        Prepares bounding box coordinates and detection ids for a given group.
        """
        all_bbox_coords = []
        detection_ids = []

        for _, row in group.iterrows():
            if pd.isna(row[1]):
                continue  # Skip if detection ID is NaN
            detection_id = int(row[1])
            bbox_coords = row[2:6].tolist()  # xmin, ymin, xmax, ymax
            detection_ids.append(detection_id)
            all_bbox_coords.append(bbox_coords)

        return all_bbox_coords, detection_ids

    def save_output(self, output_data, output_csv_path):
        """
        Save the processed output to a CSV file.
        """
        keypoints_headers = [f'hpe_keypoint_{i}_{axis}' for i in range(17) for axis in ['x', 'y']]
        output_df = pd.DataFrame(output_data, columns=['frame_number', 'detection_id', 'xmin', 'ymin', 'xmax', 'ymax'] + keypoints_headers)
        output_df.to_csv(output_csv_path, index=False)


def run_pipeline(pose_model, img_size, conf_thres, iou_thres, detections_folder, frames_folder, output_folder, subfolders_to_process):
    """
    Function to initiate the pose prediction pipeline with provided parameters.
    """
    pipeline = PosePredictionPipeline(pose_model, img_size, conf_thres, iou_thres)
    original_size = (1280, 720)
    target_size = (640, 640)

    pipeline.process_event(detections_folder, frames_folder, output_folder, original_size, target_size, subfolders_to_process)

