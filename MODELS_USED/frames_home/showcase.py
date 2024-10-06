import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import layers, models, optimizers, Input
import json
from tensorflow.keras.callbacks import ModelCheckpoint
import keras
import cv2
import matplotlib.pyplot as plt
import time
from tensorflow.keras.utils import plot_model



# Function to apply green filtering
def filter_green_shades(image, lower_bound=(0, 0.275, 0), upper_bound=(0.392, 1.0, 0.392)):
    lower_bound = tf.constant(lower_bound, dtype=tf.float32)
    upper_bound = tf.constant(upper_bound, dtype=tf.float32)
    
    # Create mask for green shades
    mask = tf.reduce_all(tf.logical_and(image >= lower_bound, image <= upper_bound), axis=-1)
    mask = tf.expand_dims(mask, axis=-1)

    # Invert the mask: 1 for non-green pixels, 0 for green pixels
    inverted_mask = tf.logical_not(mask)
    
    # Replace green pixels with black
    filtered_image = tf.where(mask, tf.zeros_like(image), image)
    
    return filtered_image, inverted_mask


# def compute_color_histogram(image, mask, num_bins=256):
#     # Convert image to float32 for histogram computation
#     image = tf.cast(image, tf.float32)
    
#     # Flatten the image and mask
#     image_flat = tf.reshape(image, [-1, 3])
#     mask_flat = tf.reshape(mask, [-1])
#     # Debug: Print shapes and initial values
#     # tf.print("Image shape:", tf.shape(image))
#     # tf.print("Mask shape:", tf.shape(mask))
#     # tf.print("Image flat shape:", tf.shape(image_flat))
#     # tf.print("Mask flat shape:", tf.shape(mask_flat))
#     #tf.print("Initial image values:", image_flat, summarize=30)
#     #tf.print("Initial mask values:", mask_flat, summarize=30)

    
#     # Use the mask to exclude filtered pixels
#     masked_image_flat = tf.boolean_mask(image_flat, mask_flat)
#     # Debug: Print masked image values
#     #tf.print("Masked image values:", masked_image_flat, summarize=30)

#     hist = []
#     for channel in range(3):  # Assuming image is RGB
#         channel_data = masked_image_flat[:, channel]
#         #tf.print('channel_data:')
#         #tf.print(channel_data)
#         channel_data = tf.cast(channel_data * 255, tf.int32)  # Scale to 0-255
#         #tf.print('channel_data_2:')
#         #tf.print(channel_data)
#         hist_channel = tf.histogram_fixed_width(channel_data, [0, 255], nbins=num_bins)
#         hist.append(hist_channel)

#     hist = tf.concat(hist, axis=0)
#     #print('hier de hist:')
#     #tf.print("Histogram values:", hist)
#     return hist


def compute_color_histogram(image, mask=None, num_bins=256):
    """
    Compute the color histogram of an image. If a mask is provided, compute the histogram for the masked image.
    
    Args:
        image (tf.Tensor): The input image tensor (H, W, C) with values in the range [0, 1].
        mask (tf.Tensor): Optional binary mask (H, W, 1) where `True` values indicate the pixels to be considered.
        num_bins (int): Number of bins to use for the histogram.
    
    Returns:
        tf.Tensor: Concatenated histogram for each channel (shape: [num_bins * 3]).
    """
    image = tf.cast(image, tf.float32)
    image_flat = tf.reshape(image, [-1, 3])
    
    if mask is not None:
        # Flatten the mask and use it to exclude filtered pixels
        mask_flat = tf.reshape(mask, [-1])
        image_flat = tf.boolean_mask(image_flat, mask_flat)
    
    hist = []
    for channel in range(3):  # Assuming image is RGB
        channel_data = image_flat[:, channel]
        channel_data = tf.cast(channel_data * 255, tf.int32)  # Scale to 0-255
        hist_channel = tf.histogram_fixed_width(channel_data, [0, 255], nbins=num_bins)
        hist.append(hist_channel)

    hist = tf.concat(hist, axis=0)
    return hist

def save_tensor_image(tensor, filename):
    tensor = tf.image.convert_image_dtype(tensor, tf.uint8)
    encoded_image = tf.image.encode_png(tensor)
    tf.io.write_file(filename, encoded_image)


# Function to plot and save histogram
def save_histogram(hist, filename, num_bins=256):
    plt.figure()
    colors = ['red', 'green', 'blue']
    histograms = [hist[0:num_bins], hist[num_bins:2*num_bins], hist[2*num_bins:3*num_bins]]
    
    for i in range(3):
        plt.plot(histograms[i], color=colors[i])
    
    plt.title('Color Histogram')
    plt.xlabel('Bin')
    plt.ylabel('Frequency')
    plt.savefig(filename)
    plt.close()

# def decode_resize_and_histogram(file_path, img_height=224, img_width=224):
#     img = tf.io.read_file(file_path)
#     save_tensor_image(tf.image.decode_jpeg(img, channels=3), os.path.join('/home/ivanmiert/frames_home/showcase', 'eerste.png'))
#     img = tf.image.decode_jpeg(img, channels=3)
#     save_tensor_image(img, os.path.join('/home/ivanmiert/frames_home/showcase', 'tweede.png'))
#     img = tf.ensure_shape(img, [None, None, 3])
#     #tf.print("Step 2 - Decoded Image Range:", tf.reduce_min(img), tf.reduce_max(img))
#     img = tf.image.convert_image_dtype(img, tf.float32)
#     img = tf.image.resize(img, [img_height, img_width])
#     save_tensor_image(img, os.path.join('/home/ivanmiert/frames_home/showcase', 'derde.png'))
#     #tf.print("Step 3 - Resized Image Range:", tf.reduce_min(img), tf.reduce_max(img))
#     hist1 = compute_color_histogram(img, mask, num_bins=256)
#     save_histogram(hist1.numpy(), os.path.join('/home/ivanmiert/frames_home/showcase', 'histogram_non_filtered.png'))
#     filtered_img, mask = filter_green_shades(img)  # Apply green filtering and get mask
#     save_tensor_image(filtered_img, os.path.join('/home/ivanmiert/frames_home/showcase', 'filtered.png'))
#     #tf.print("Step 4 - Filtered Image Range:", tf.reduce_min(filtered_img), tf.reduce_max(filtered_img))
#     #print('hij komt wel hier:')
#     hist = compute_color_histogram(filtered_img, mask, num_bins=256)
#     save_histogram(hist.numpy(), os.path.join('/home/ivanmiert/frames_home/showcase', 'histogram_filtered.png'))

def decode_resize_and_histogram(file_path, img_height=224, img_width=224):
    """
    Decode an image from file, resize, filter green shades, and compute histograms for both the original
    and filtered images.
    
    Args:
        file_path (str): Path to the image file.
        img_height (int): Desired height of the resized image.
        img_width (int): Desired width of the resized image.
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.ensure_shape(img, [None, None, 3])
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    
    # Save original image for debugging
    save_tensor_image(img, os.path.join('/home/ivanmiert/frames_home/showcase', 'original.png'))

    # Compute histogram for the original image
    hist_original = compute_color_histogram(img)
    
    # Save histogram for the original image
    save_histogram(hist_original.numpy(), os.path.join('/home/ivanmiert/frames_home/showcase', 'histogram_original.png'))

    # Apply green filtering and get mask
    filtered_img, mask = filter_green_shades(img)
    
    # Save filtered image for debugging
    save_tensor_image(filtered_img, os.path.join('/home/ivanmiert/frames_home/showcase', 'filtered.png'))

    # Compute histogram for the filtered image
    hist_filtered = compute_color_histogram(filtered_img, mask)
    
    # Save histogram for the filtered image
    save_histogram(hist_filtered.numpy(), os.path.join('/home/ivanmiert/frames_home/showcase', 'histogram_filtered.png'))

#file_path = '/scratch-shared/ivanmiert/classifierdata_2_kopie_2/udinese_Keeper1/0_crop_6.jpg'
file_path = '/scratch-shared/ivanmiert/classifierdata_2_kopie_2/sassuolo_1/1_crop_23.jpg'
decode_resize_and_histogram(file_path)