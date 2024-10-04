import sys 
sys.path.insert(0, '/home/ivanmiert/NEW_OBJECT/CLASSIFICATION_MODEL')

from create_label_overview import create_label_and_length_dataframe
import yaml

# Load configuration from YAML file
def load_config(config_path='/home/ivanmiert/overview/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Example usage
config = load_config()  # Assuming config.yaml is in the same directory

designated_events_folder_features = config['template_folders']['concatenated_features_bottom_up']
dataframe = config['event_data']['all']
output_path = config['output_files']['labels_all']

create_label_and_length_dataframe(designated_events_folder_features, dataframe, output_path)