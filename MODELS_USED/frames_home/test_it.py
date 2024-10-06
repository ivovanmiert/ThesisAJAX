import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.applications import vgg16
from tensorflow import keras
import matplotlib.pyplot as plt

class ImageClassifier:
    def __init__(self, model_path, class_names_path):
        # Load the model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load class names
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        
        # Define image size and number of bins for histograms
        self.img_height = 224
        self.img_width = 224
        self.num_bins = 256
    
    #Function to compute color histogram
    def compute_color_histogram(self, image, mask, num_bins=256):
        hist = []
        for channel in range(3):  # Assuming image is RGB
            masked_image = tf.boolean_mask(image[:, :, channel], ~tf.squeeze(mask))
            hist_channel = tf.histogram_fixed_width(masked_image, [0, 256], nbins=num_bins)
            hist.append(hist_channel)
        hist = tf.concat(hist, axis=0)
        return hist
        
    def preprocess_image(self, image_path):
        """Preprocess the input image for prediction, including green filtering."""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        
        # Apply green filtering
        img, mask = self.filter_green_shades(img)  # Apply the green filtering
        
        hist = self.compute_color_histogram(img, mask, 256)

        img = vgg16.preprocess_input(img)  # Apply the same preprocessing used during training
        img = tf.expand_dims(img, 0)  # Add batch dimension
        #hist = self.compute_color_histogram(img[0], mask) 
        return img, hist
    
    def filter_green_shades(self, image, lower_bound=(0, 50, 0), upper_bound=(100, 255, 100)):
        """Filter green shades from the image."""
        lower_bound = tf.constant(lower_bound, dtype=tf.float32)
        upper_bound = tf.constant(upper_bound, dtype=tf.float32)
        
        # Create mask for green shades
        mask = tf.reduce_all(tf.logical_and(image >= lower_bound, image <= upper_bound), axis=-1)
        mask = tf.expand_dims(mask, axis=-1)
        
        # Replace green pixels with black
        filtered_image = tf.where(mask, tf.zeros_like(image), image)
        plt.figure(figsize=(15,10))
        plt.title("Filtered Image")
        plt.imshow(filtered_image.numpy().astype("uint8"))
        plt.axis('off')
        plt.savefig('/home/ivanmiert/frames_home/figure_filtered.png')
        
        return filtered_image, mask



    def predict(self, image_path, top_classes, top_n=5):
        """Predict the class of the input image and return the top N predictions among specified classes."""
        # Preprocess the image
        img, hist = self.preprocess_image(image_path)
        plt.figure(figsize=(15,10))
        plt.title("Filtered Image Histogram")
        for i, color in enumerate(['r', 'g', 'b']):
            plt.plot(hist[i * 256:(i + 1) * 256], color=color)
        plt.xlim([0, 256])
        plt.savefig('/home/ivanmiert/frames_home/figure_filtered_hist.png')

        # Perform prediction
        predictions = self.model.predict([img, tf.expand_dims(hist,0)])
        
        # Get the top class indices
        class_indices = [self.class_names.index(c) for c in top_classes]
        
        # Filter predictions to include only the top classes
        filtered_preds = predictions[0][class_indices]
        
        # Get the index of the maximum prediction from the filtered predictions
        max_index = tf.argmax(filtered_preds)
        
        # Get the top prediction and its probability
        top_class_index = class_indices[max_index.numpy()]
        top_class_name = self.class_names[top_class_index]
        top_prob = filtered_preds[max_index.numpy()]
        
        return top_class_name, top_prob
    
    def print_predictions(self, image_path, top_classes, top_n=5):
        """Print the top prediction among specified classes for the input image."""
        top_class_name, top_prob = self.predict(image_path, top_classes, top_n)
        
        print(f"Top prediction for '{image_path}':")
        print(f"Class: {top_class_name}, Probability: {top_prob:.4f}")

# Example usage:
if __name__ == "__main__":
    # Define paths to your saved model and class names file
    model_path = '/home/ivanmiert/frames_home/football_classifier_model_vgg16_with_hist_manual_matrix_001_11classes_filtered.keras'
    class_names_path = '/home/ivanmiert/frames_home/class_names_11.json'
    
    # Create an instance of the classifier
    classifier = ImageClassifier(model_path, class_names_path)
    
    # Path to the image you want to classify
    image_path = "/home/ivanmiert/frames_home/penalty_classify/Test_BB/second_test.png"
    
    # List of possible classes for the model to choose from
    possible_classes = ['Yellow', 'White', 'Black', 'Purple', 'Green']
    
    # Print the top prediction among the specified classes
    classifier.print_predictions(image_path, possible_classes, top_n=5)