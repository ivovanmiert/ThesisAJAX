import numpy as np
from keras import preprocessing
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input
import os

"""
This file creates the Team Classifier object. For every event such a Team Classifier object is made, which gets as input:
    - kits: 5 kits for that event
    - classify_model: the model trained on all kits

The Team Classifier is instantiated at the start of the handling of the event. 
During the event, it is used when player instances are instantiated. 
It is used to classify the player into 1 of 5 classes. 


"""



class TeamClassifier:
    def __init__(self, kits, classify_model, save_dir):
        self.kits = kits
        self.model = classify_model.get_model()
        self.class_names = classify_model.get_class_names()
        self.class_names_dict = {name: idx for idx, name in enumerate(self.class_names)}
        self.possible_kits = [
            self.kits['kit_team_1'],
            self.kits['kit_team_2'],
            self.kits['keeper_kit_team_1'],
            self.kits['keeper_kit_team_2'],
            self.kits['referee_kit']
        ]

        self.possible_label_indices = [self.class_names_dict[label] for label in self.possible_kits]
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def compute_color_histogram(self, image, mask, num_bins=256):
        # Convert image to float32 for histogram computation
        image = tf.cast(image, tf.float32)
        image_flat = tf.reshape(image, [-1, 3])
        mask_flat = tf.reshape(mask, [-1])
        inverted_mask_flat = tf.logical_not(mask_flat)
        masked_image_flat = tf.boolean_mask(image_flat, inverted_mask_flat)

        hist = []
        for channel in range(3):  # Assuming image is RGB
            channel_data = masked_image_flat[:, channel]
            channel_data = tf.cast(channel_data * 255, tf.int32)  # Scale to 0-255
            hist_channel = tf.histogram_fixed_width(channel_data, [0, 255], nbins=num_bins)
            hist.append(hist_channel)

        hist = tf.concat(hist, axis=0)
        return hist

    def filter_green_shades(self, image, player_id, lower_bound=(0, 0.275, 0), upper_bound=(0.392, 1.0, 0.392)):
        lower_bound = tf.constant(lower_bound, dtype=tf.float32)
        upper_bound = tf.constant(upper_bound, dtype=tf.float32)

        # Create mask for green shades
        mask = tf.reduce_all(tf.logical_and(image >= lower_bound, image <= upper_bound), axis=-1)
        mask = tf.expand_dims(mask, axis=-1)

        # Replace green pixels with black
        filtered_image = tf.where(mask, tf.zeros_like(image), image)


        # Convert filtered image to uint8 and save
        filtered_image_uint8 = tf.cast(filtered_image * 255, tf.uint8)
        filtered_image_np = filtered_image_uint8.numpy()
        return filtered_image, mask

    def preprocess_input_image(self, cropped_bounding_box, player_id):
        img = tf.convert_to_tensor(cropped_bounding_box, dtype=tf.float32)
        img = img/255.0
        img = tf.image.resize(img, [224, 224])
        filtered_img, mask = self.filter_green_shades(img, player_id)
        # Debug: Check the range of values in filtered image
        hist = self.compute_color_histogram(filtered_img, mask, num_bins=256)
        # Debug: Check the histogram values
        preprocessed_image = preprocess_input(filtered_img)
        return preprocessed_image, hist

    def classify(self, bounding_box, frame_id, frames_object, player_id, filename_prefix="output"):
        img = frames_object.get_frame(frame_id)
        if img is None:
            raise ValueError(f"Frame {frame_id} not found in {frames_object.clip_folder_path}")
        x_min, y_min, x_max, y_max = bounding_box
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)
        cropped_bounding_box = img[y_min:y_max, x_min:x_max]
        
        pre_processed_img, histogram = self.preprocess_input_image(cropped_bounding_box, player_id)

        pre_processed_img = tf.expand_dims(pre_processed_img, axis=0)  # Shape: (1, 224, 224, 3)
        histogram = tf.expand_dims(histogram, axis=0)  # Shape: (1, 768)
        histogram_plot = histogram[0]
        histogram_1 = histogram_plot[0:256]
        histogram_2 = histogram_plot[256:512]
        histogram_3 = histogram_plot[512:768]
        prediction = self.model.predict([pre_processed_img, histogram])
        filtered_prediction = prediction[:, self.possible_label_indices]
        predicted_index = np.argmax(filtered_prediction, axis=1)[0]
        predicted_class = self.possible_kits[predicted_index]
        if predicted_class == self.kits['kit_team_1']:
            return 'Team_1'
        elif predicted_class == self.kits['kit_team_2']:
            return 'Team_2'
        elif predicted_class == self.kits['keeper_kit_team_1']:
            return 'Goalkeeper_Team_1'
        elif predicted_class == self.kits['keeper_kit_team_2']:
            return 'Goalkeeper_Team_2'
        elif predicted_class == self.kits['referee_kit']:
            return 'Referee'
        else:
            return 'Unknown'



        