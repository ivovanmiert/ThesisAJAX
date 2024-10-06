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

# Clear all previously registered custom objects
keras.saving.get_custom_objects().clear()

penalty_matrix = pd.read_csv('/home/ivanmiert/frames_home/penalty_classify/penalty_matrix_manual_03_corrected.csv', index_col=0)
penalty_matrix_np = penalty_matrix.to_numpy()
print(penalty_matrix.shape)
print(penalty_matrix_np)

@keras.saving.register_keras_serializable(package='Custom', name='custom_categorical_crossentropy_with_penalty')
def custom_categorical_crossentropy_with_penalty(y_true, y_pred):
    penalty_matrix = tf.constant(penalty_matrix_np, dtype=tf.float32)


    true_indices = tf.argmax(y_true, axis=-1)


    cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)


    penalties = tf.gather_nd(penalty_matrix, tf.stack([true_indices, tf.argmax(y_pred, axis=-1)], axis=1))

    total_loss = cce_loss * penalties

    return total_loss


class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, class_names, num_samples=5, save_dir='output', target_classes=[16, 33]):
        super(PredictionCallback, self).__init__()
        self.dataset = dataset
        self.class_names = class_names
        self.num_samples = num_samples
        self.save_dir = save_dir
        self.total_epochs = 15
        os.makedirs(self.save_dir, exist_ok=True)
        self.target_classes = target_classes if target_classes is not None else []

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 == self.total_epochs:
            print(f"\nEpoch {epoch + 1}: Sample Predictions and Saving Filtered Images and Histograms")
            for inputs, true_labels in self.dataset.take(self.num_samples):
                images, histograms = inputs
                predictions = self.model.predict((images, histograms))
                predicted_labels = tf.argmax(predictions, axis=-1)
                true_labels = tf.argmax(true_labels, axis=-1)

                print(f"Image shape: {images.shape}, type: {type(images)}, dtype: {images.dtype}")
                print(f"Histogram shape: {histograms.shape}, type: {type(histograms)}, dtype: {histograms.dtype}")
                print(f"Sample image data (first image): {images[0].numpy()[:5][:5][:5]}")  # First 5 values of the first 5x5 pixels
                print(f"Sample histogram data (first histogram): {histograms[0].numpy()[:767]}")  # First 10 values of the first histogram

                for i in range(len(predicted_labels)):
                    if true_labels[i].numpy() in self.target_classes or predicted_labels[i].numpy() in self.target_classes:
                        self.save_filtered_image_and_histogram(images[i], histograms[i], true_labels[i], predicted_labels[i], i)

                for i in range(len(predicted_labels)):
                    print(f"True label: {self.class_names[true_labels[i]]}, Predicted label: {self.class_names[predicted_labels[i]]}, predictions: {predictions[i]}")

    def save_filtered_image_and_histogram(self, image, histogram, true_label, predicted_label, sample_idx):
        histogram_1 = histogram[0:256]
        histogram_2 = histogram[256:512]
        histogram_3 = histogram[512:768]

        plt.figure()
        colors = ['red', 'green', 'blue']
        histograms = [histogram_1, histogram_2, histogram_3]

        for i in range(3):
            plt.plot(histograms[i], color=colors[i])

        plt.title(f'True: {self.class_names[true_label]}, Pred: {self.class_names[predicted_label]}')
        plt.savefig(os.path.join(self.save_dir, f'sample_{sample_idx}_histogram_test.png'))
        plt.close()



batch_size = 32
img_height = 224
img_width = 224
num_bins = 256
max_samples_per_class = 50
def get_dataset_info(directory):
    file_paths = []
    labels = []
    class_names = sorted(os.listdir(directory))

    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            all_files = [os.path.join(class_dir, file_name) for file_name in os.listdir(class_dir)]
            # Sample a maximum of 200 files per class
            sampled_files = np.random.choice(all_files, min(len(all_files), max_samples_per_class), replace=False)
            file_paths.extend(sampled_files)
            labels.extend([class_name] * len(sampled_files))

    return pd.DataFrame({'file_path': file_paths, 'label': labels}), class_names


def filter_green_shades(image, lower_bound=(0, 0.275, 0), upper_bound=(0.392, 1.0, 0.392)):
    lower_bound = tf.constant(lower_bound, dtype=tf.float32)
    upper_bound = tf.constant(upper_bound, dtype=tf.float32)


    mask = tf.reduce_all(tf.logical_and(image >= lower_bound, image <= upper_bound), axis=-1)
    mask = tf.expand_dims(mask, axis=-1)


    inverted_mask = tf.logical_not(mask)

    filtered_image = tf.where(mask, tf.zeros_like(image), image)

    return filtered_image, inverted_mask

pd.options.display.max_rows = 150
pd.options.display.max_columns = 150
# Load dataset info
dataset_dir = '/scratch-shared/ivanmiert/classifierdata_2_kopie_2'
df, class_names = get_dataset_info(dataset_dir)
# Encode labels to integers
df['label'] = df['label'].astype('category').cat.codes
class_counts = df['label'].value_counts()
#print(class_counts)
# Split dataset into stratified train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'])

def compute_color_histogram(image, mask, num_bins=256):

    image = tf.cast(image, tf.float32)

    # Flatten the image and mask
    image_flat = tf.reshape(image, [-1, 3])
    mask_flat = tf.reshape(mask, [-1])

    # tf.print("Image shape:", tf.shape(image))
    # tf.print("Mask shape:", tf.shape(mask))
    # tf.print("Image flat shape:", tf.shape(image_flat))
    # tf.print("Mask flat shape:", tf.shape(mask_flat))
    #tf.print("Initial image values:", image_flat, summarize=30)
    #tf.print("Initial mask values:", mask_flat, summarize=30)


    # Use the mask to exclude filtered pixels
    masked_image_flat = tf.boolean_mask(image_flat, mask_flat)
    # Debug: Print masked image values
    #tf.print("Masked image values:", masked_image_flat, summarize=30)

    hist = []
    for channel in range(3):  # Assuming image is RGB
        channel_data = masked_image_flat[:, channel]
        #tf.print('channel_data:')
        #tf.print(channel_data)
        channel_data = tf.cast(channel_data * 255, tf.int32)  # Scale to 0-255
        #tf.print('channel_data_2:')
        #tf.print(channel_data)
        hist_channel = tf.histogram_fixed_width(channel_data, [0, 255], nbins=num_bins)
        hist.append(hist_channel)

    hist = tf.concat(hist, axis=0)
    #print('hier de hist:')
    #tf.print("Histogram values:", hist)
    return hist

def save_tensor_image(tensor, filename):
    tensor = tf.image.convert_image_dtype(tensor, tf.uint8)
    encoded_image = tf.image.encode_png(tensor)
    tf.io.write_file(filename, encoded_image)


# Function to decode image, resize and compute histogram
def decode_resize_and_histogram(file_path, label, img_height=224, img_width=224):
    img = tf.io.read_file(file_path)
    save_tensor_image(tf.image.decode_jpeg(img, channels=3), os.path.join('/home/ivanmiert/frames_home/penalty_classify/examples', 'eerste.png'))
    img = tf.image.decode_jpeg(img, channels=3)
    save_tensor_image(img, os.path.join('/home/ivanmiert/frames_home/penalty_classify/examples', 'tweede.png'))
    img = tf.ensure_shape(img, [None, None, 3])
    #tf.print("Step 2 - Decoded Image Range:", tf.reduce_min(img), tf.reduce_max(img))
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    save_tensor_image(img, os.path.join('/home/ivanmiert/frames_home/penalty_classify/examples', 'derde.png'))
    #tf.print("Step 3 - Resized Image Range:", tf.reduce_min(img), tf.reduce_max(img))
    filtered_img, mask = filter_green_shades(img)  # Apply green filtering and get mask
    save_tensor_image(filtered_img, os.path.join('/home/ivanmiert/frames_home/penalty_classify/examples', 'filtered.png'))
    #tf.print("Step 4 - Filtered Image Range:", tf.reduce_min(filtered_img), tf.reduce_max(filtered_img))
    #print('hij komt wel hier:')
    hist = compute_color_histogram(filtered_img, mask, num_bins=256)

    preprocessed_image = preprocess_input(filtered_img)  # Preprocess input for VGG16
    label = tf.cast(label, tf.int32)
    return preprocessed_image, hist, label

# # Function to one-hot encode labels
def one_hot_encode_labels(labels, num_classes):
    return tf.one_hot(labels, depth=num_classes)

# Function to load data from DataFrame with one-hot encoding
def load_data_with_histograms(df, num_classes):
    file_paths = df['file_path'].values
    labels = df['label'].values
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(decode_resize_and_histogram, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda preprocessed_image, hist, label: ((preprocessed_image, hist), one_hot_encode_labels(label, num_classes)))
    return dataset

# Create TensorFlow datasets
train_dataset = load_data_with_histograms(train_df, num_classes=len(class_names))
val_dataset = load_data_with_histograms(val_df, num_classes=len(class_names))

# Custom brightness adjustment function
def random_brightness(image, max_delta=0.2):
    return tf.image.random_brightness(image, max_delta=max_delta)

rotation_layer = layers.RandomRotation(0.05)
contrast_layer = layers.RandomContrast(0.05)

# Correct combined_augmentation function to accept and return all required arguments
def combined_augmentation(inputs, label):
    image, hist = inputs
    #tf.print("Histogram values in combined_augmentation:", hist)
    image = tf.expand_dims(image, 0)
    image = rotation_layer(image, training=True)  # Use the pre-defined layer
    image = contrast_layer(image, training=True)  # Use pre-defined layer
    image = random_brightness(image)
    image = tf.squeeze(image, 0)  # Remove the batch dimension
    return (image, hist), label

# Apply data augmentation to the training dataset
train_dataset = train_dataset.map(combined_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle and batch the datasets
train_dataset = train_dataset.shuffle(buffer_size=len(train_df)).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
# for (images, hist), label in train_dataset.take(1):
#   #print('images: ', images)
#   #print('hist', hist)
#   #Histograms zien er hier vrij goed uit, dus dat is mooi. Nu nog images checken
#   #print('label:', label)


def create_model_with_histogram(input_shape_image, input_shape_histogram, num_classes):
    image_input = Input(shape=input_shape_image)
    #hist_input = Input(shape=input_shape_histogram)

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape_image)
    # Set all layers to non-trainable by default
    for layer in base_model.layers:
        layer.trainable = False

    #Make the last layers trainable
    for layer in base_model.layers[-8:]:
        layer.trainable = True

    x = base_model(image_input, training=False)
    x = layers.Flatten()(x)

    # Histogram Input Branch
    hist_input = Input(shape=input_shape_histogram)
    #print("In the training loop:", hist_input)
    y = layers.Dense(256, activation='relu')(hist_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dense(256, activation='relu')(y)
    y = layers.Dropout(0.5)(y)  # Higher dropout to prevent overfitting
    y = layers.Dense(128, activation='relu')(y)
    # Scale the processed histogram features
    scale_factor = 10.0  # Adjust this factor as needed
    scaled_hist_input = layers.Lambda(lambda x: x * scale_factor)(y)

    combined_features = layers.concatenate([x, scaled_hist_input])
    combined_features = layers.Dense(512, activation='relu')(combined_features)
    combined_features = layers.Dropout(0.2)(combined_features)  # Add dropout for regularization
    output = layers.Dense(num_classes, activation='softmax')(combined_features)

    model = models.Model(inputs=[image_input, hist_input], outputs=output)

    return model

# Model input shapes
input_shape_image = (img_height, img_width, 3)
input_shape_histogram = (3 * num_bins,)
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")


model = create_model_with_histogram(input_shape_image, input_shape_histogram, num_classes)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=custom_categorical_crossentropy_with_penalty, metrics=['accuracy'])
print('learning_rate=0.001')


prediction_callback = PredictionCallback(val_dataset, class_names, num_samples=5, save_dir='/home/ivanmiert/frames_home/penalty_classify/examples', target_classes=[16, 33])

# checkpoint_path = '/home/ivanmiert/frames_home/best_model.keras.h5'
# model_checkpoint_callback = ModelCheckpoint(
#     filepath=checkpoint_path,
#     save_weights_only=False,
#     monitor='val_accuracy',
#     mode='max',
#     save_best_only=True,
#     verbose=1
# )

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    callbacks=[prediction_callback]
)


loss, accuracy = model.evaluate(val_dataset)
print(f'Validation accuracy: {accuracy * 100:.2f}%')


model.save('/home/ivanmiert/frames_home/football_classifier_model_vgg16_all_classes_filtered_4layers_plusscaledhisto_new_05_14_39-second.keras')
with open('/home/ivanmiert/frames_home/class_names_all.json', 'w') as f:
    json.dump(class_names, f)

plot_model(model, to_file='/home/ivanmiert/frames_home/model_architecture.png', show_shapes=True, show_layer_names=True)
