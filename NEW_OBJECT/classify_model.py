import tensorflow as tf
from tensorflow import keras
import json
import pandas as pd
import tensorflow.keras.backend as K
import keras


"""
This file is used to load the classification model from its saved path. 
It is then used later in the 'classify.py' file to be loaded for each TeamClassifier object
"""

penalty_matrix = pd.read_csv('/home/ivanmiert/frames_home/penalty_classify/penalty_matrix_manual_03_corrected.csv', header=None)
penalty_matrix_np = penalty_matrix.to_numpy()




# Define a custom loss function
@keras.saving.register_keras_serializable(package='Custom', name='custom_categorical_crossentropy_with_penalty')
def custom_categorical_crossentropy_with_penalty(y_true, y_pred):
    penalty_matrix = tf.constant(penalty_matrix_np, dtype=tf.float32)

    # Convert true labels from one-hot encoding to indices
    true_indices = tf.argmax(y_true, axis=-1)

    # Compute categorical cross-entropy loss
    cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    # Compute penalties for misclassifsication
    penalties = tf.gather_nd(penalty_matrix, tf.stack([true_indices, tf.argmax(y_pred, axis=-1)], axis=1))

    # Add the penalties to the cross-entropy loss
    total_loss = cce_loss * penalties

    return total_loss

class ClassifyModel:
    def __init__(self, model_path, class_names_path):
        """
        Initializes the ClassifyModel with the provided model path.
        
        :param model_path: Path to the pre-trained model (.keras file).
        """
        # Load penalty matrix from a CSV file
        #Hier bij aanpassen:
        
        num_classes = 132
        with open(class_names_path, 'r') as file:
            self.class_names = json.load(file)
        class_name_to_index = {name: index for index, name in enumerate(self.class_names)}

        
        self.model = tf.keras.models.load_model(model_path, safe_mode=False) #, custom_objects={'custom_loss': custom_categorical_crossentropy_with_penalty}, safe_mode=False)    

    def get_model(self):
        #Returns the loaded model.
        return self.model
    
    def get_class_names(self):
        #Returns the class names the model was trained on 

        return self.class_names