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


class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, class_names, num_samples=5):
        super(PredictionCallback, self).__init__()
        self.dataset = dataset
        self.class_names = class_names
        self.num_samples = num_samples

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1}: Sample Predictions")
        for inputs, true_labels in self.dataset.take(self.num_samples):
            images, histograms = inputs
            predictions = self.model.predict((images, histograms))
            predicted_labels = tf.argmax(predictions, axis=-1)
            true_labels = tf.argmax(true_labels, axis=-1)
            for i in range(len(predicted_labels)):
                print(f"True label: {self.class_names[true_labels[i]]}, Predicted label: {self.class_names[predicted_labels[i]]}, predictions:{predictions}")

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
            sampled_files = np.random.choice(all_files, min(len(all_files), max_samples_per_class), replace=False)
            file_paths.extend(sampled_files)
            labels.extend([class_name] * len(sampled_files))

    return pd.DataFrame({'file_path': file_paths, 'label': labels}), class_names

def filter_green_shades(image, lower_bound=(0, 70, 0), upper_bound=(100, 255, 100)):
    lower_bound = tf.constant(lower_bound, dtype=tf.float32)
    upper_bound = tf.constant(upper_bound, dtype=tf.float32)

    mask = tf.reduce_all(tf.logical_and(image >= lower_bound, image <= upper_bound), axis=-1)
    mask = tf.expand_dims(mask, axis=-1)

    filtered_image = tf.where(mask, tf.zeros_like(image), image)

    return filtered_image, mask

pd.options.display.max_rows = 150
pd.options.display.max_columns = 150
dataset_dir = '/scratch-shared/ivanmiert/classifierdata_2'
df, class_names = get_dataset_info(dataset_dir)
df['label'] = df['label'].astype('category').cat.codes
class_counts = df['label'].value_counts()
print(class_counts)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=43)

def compute_color_histogram(image, mask, num_bins=256):
    image = tf.cast(image, tf.float32)

    image_flat = tf.reshape(image, [-1, 3])
    mask_flat = tf.reshape(mask, [-1])

    masked_image_flat = tf.boolean_mask(image_flat, mask_flat)

    hist = []
    for channel in range(3):
        channel_data = masked_image_flat[:, channel]
        hist_channel = tf.histogram_fixed_width(channel_data, [0, 256], nbins=num_bins)
        hist.append(hist_channel)

    hist = tf.concat(hist, axis=0)
    return hist

def decode_resize_and_histogram(file_path, label, img_height=224, img_width=224):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = preprocess_input(img)  # Preprocess input for VGG16
    filtered_img, mask = filter_green_shades(img)  # Apply green filtering and get mask
    hist = compute_color_histogram(filtered_img, mask, num_bins)
    label = tf.cast(label, tf.int32)
    return filtered_img, hist, label


def one_hot_encode_labels(labels, num_classes):
    return tf.one_hot(labels, depth=num_classes)

def load_data_with_histograms(df, num_classes):
    file_paths = df['file_path'].values
    labels = df['label'].values
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(decode_resize_and_histogram, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda img, hist, label: ((img, hist), one_hot_encode_labels(label, num_classes)))
    return dataset

train_dataset = load_data_with_histograms(train_df, num_classes=len(class_names))
val_dataset = load_data_with_histograms(val_df, num_classes=len(class_names))


def random_brightness(image, max_delta=0.2):
    return tf.image.random_brightness(image, max_delta=max_delta)

rotation_layer = layers.RandomRotation(0.1)
contrast_layer = layers.RandomContrast(0.2)


def combined_augmentation(inputs, label):
    image, hist = inputs
    image = tf.expand_dims(image, 0)
    image = rotation_layer(image, training=True)  # Use the pre-defined layer
    image = contrast_layer(image, training=True)  # Use pre-defined layer
    image = random_brightness(image)
    image = tf.squeeze(image, 0)  # Remove the batch dimension
    return (image, hist), label

train_dataset = train_dataset.map(combined_augmentation, num_parallel_calls=tf.data.AUTOTUNE)


train_dataset = train_dataset.shuffle(buffer_size=len(train_df)).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)


def create_model_with_histogram(input_shape_image, input_shape_histogram, num_classes):
    image_input = Input(shape=input_shape_image)
    hist_input = Input(shape=input_shape_histogram)

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape_image)
    # Set all layers to non-trainable by default
    for layer in base_model.layers:
        layer.trainable = True

    #Make the last layers trainable
    for layer in base_model.layers[-12:]:
        layer.trainable = True

    x = base_model(image_input, training=False)
    x = layers.Flatten()(x)

    combined_features = layers.concatenate([x, hist_input])
    combined_features = layers.Dense(512, activation='relu')(combined_features)
    combined_features = layers.Dropout(0.2)(combined_features)  # Add dropout for regularization
    output = layers.Dense(num_classes, activation='softmax')(combined_features)

    model = models.Model(inputs=[image_input, hist_input], outputs=output)

    return model

input_shape_image = (img_height, img_width, 3)
input_shape_histogram = (3 * num_bins,)
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")

penalty_matrix = pd.read_csv('/home/ivanmiert/frames_home/penalty_classify/penalty_matrix_manual_08_new.csv', header=None)
penalty_matrix_np = penalty_matrix.to_numpy()
print(penalty_matrix.shape)
print(penalty_matrix_np)
def custom_categorical_crossentropy_with_penalty(y_true, y_pred):
    penalty_matrix = tf.constant(penalty_matrix_np, dtype=tf.float32)


    true_indices = tf.argmax(y_true, axis=-1)


    cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)


    penalties = tf.gather_nd(penalty_matrix, tf.stack([true_indices, tf.argmax(y_pred, axis=-1)], axis=1))

    total_loss = cce_loss * penalties

    return total_loss



model = create_model_with_histogram(input_shape_image, input_shape_histogram, num_classes)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=custom_categorical_crossentropy_with_penalty, metrics=['accuracy'])
print('learning_rate=0.001')


prediction_callback = PredictionCallback(val_dataset, class_names, num_samples=5)
checkpoint_path = '/home/ivanmiert/frames_home/best_model.keras'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    callbacks=[prediction_callback, model_checkpoint_callback]
)

loss, accuracy = model.evaluate(val_dataset)
print(f'Validation accuracy: {accuracy * 100:.2f}%')


model.save('/home/ivanmiert/frames_home/football_classifier_model_vgg16_all_classes_filtered_11layers_new_03.keras')
with open('/home/ivanmiert/frames_home/class_names_all.json', 'w') as f:
    json.dump(class_names, f)
