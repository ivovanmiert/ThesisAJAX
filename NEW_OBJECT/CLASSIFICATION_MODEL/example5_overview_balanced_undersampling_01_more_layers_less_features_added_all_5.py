import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import pandas as pd
import time

"""This file was solely meant to load the data into the different saving paths for training,valid,test. 

"""

def load_data_from_folder(folder_path, label_csv_path):
    label_df = pd.read_csv(label_csv_path)
    label_dict = label_df.set_index('clip_id').to_dict(orient='index')
    data = []
    scaler = StandardScaler()
    all_features = []
    go_time = time.time()
    print(go_time)
    i = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # i+= 1
            # if i > 100: 
            #     continue
            print(filename)
            clip_id = filename.replace('.csv', '')
            if int(clip_id) not in label_dict: #or label_dict[clip_id] not in [0, 1, 2, 3, 4]:  # Filter labels
                print('yes')
                continue

            try:
                df = pd.read_csv(os.path.join(folder_path, filename))
            except pd.errors.EmptyDataError:
                print(f"Empty or malformed CSV file: {filename}")
                continue

            df.fillna(0, inplace=True)
            features = df.drop(columns=['frame_number', 'ID']).values
            all_features.append(features)


    if all_features: 
        all_features = np.vstack(all_features)
        scaler.fit(all_features)

    start_time = time.time()
    print(start_time)
    j = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # j+= 1
            # if j > 100:
            #     continue
            clip_id = filename.replace('.csv', '')
            if int(clip_id) not in label_dict: #or label_dict[clip_id] not in [0, 1, 2, 3, 4]:  # Filter labels
                continue

            labels = label_dict[int(clip_id)]
            try:
                df = pd.read_csv(os.path.join(folder_path, filename))
            except pd.errors.EmptyDataError:
                print(f"Empty or malformed CSV file: {filename}")
                continue

            df.fillna(0, inplace=True)
            
            features_columns = df.drop(columns=['frame_number', 'ID']).columns
            df[features_columns] = scaler.transform(df[features_columns])

            frames = df['frame_number'].unique()
            frame_data = []
            for frame in frames:
                frame_df = df[df['frame_number'] == frame]
                features = frame_df.drop(columns=['frame_number', 'ID']).values
                frame_data.append({'objects': frame_df['ID'].values, 'features': features})
            
            data.append({'frame_data': frame_data, 'clip_id': clip_id, **labels})
    end_time = time.time()
    total_time = end_time - start_time
    print(f"total time: {total_time}")
    return data

# Dataset class
class ClipDataset(Dataset):
    def __init__(self, data, label_columns):
        self.data = data
        self.label_columns = label_columns
        self.scaler = StandardScaler()
        all_features = np.concatenate([frame['features'] for clip in data for frame in clip['frame_data']], axis=0)
        self.scaler.fit(all_features)
        self.label_encoders = {col: LabelEncoder() for col in label_columns}
        for col in label_columns:
            all_labels = [clip[col] for clip in data] 
            self.label_encoders[col].fit(all_labels)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip = self.data[idx]
        for frame in clip['frame_data']:
            frame['features'] = self.scaler.transform(frame['features'])

        labels = {col: torch.tensor(self.label_encoders[col].transform([clip[col]])[0], dtype=torch.long)
                  for col in self.label_columns}
        
        return {'frame_data': clip['frame_data'], 'labels': labels, 'clip_id': clip['clip_id']}
    
    def get_labels(self, label_col):
        # Return labels for a specific label column
        return [self.label_encoders[label_col].transform([clip[label_col]])[0] for clip in self.data]

    
# Collate function
def collate_fn(batch):
    frame_data_list = [item['frame_data'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    max_frames = max(len(frame_data) for frame_data in frame_data_list)
    max_objects = max(max(len(frame['objects']) for frame in frame_data) for frame_data in frame_data_list)
    feature_dim = frame_data_list[0][0]['features'].shape[1]
    
    padded_sequences = torch.zeros((len(batch), max_frames, max_objects, feature_dim))
    masks = torch.zeros((len(batch), max_frames, max_objects), dtype=torch.bool)
    
    for i, frame_data in enumerate(frame_data_list):
        for j, frame in enumerate(frame_data):
            num_objects = len(frame['objects'])
            padded_sequences[i, j, :num_objects, :] = torch.tensor(frame['features'], dtype=torch.float32)
            masks[i, j, :num_objects] = 1
    
    return padded_sequences, labels, masks


def split_dataset(dataset, val_split=0.2, test_split=0.1):
    test_size = int(len(dataset)* test_split)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size - test_size
    return random_split(dataset, [train_size, val_size, test_size])

def train_model(folder_path, label_csv_path, input_size=28, hidden_size=128, num_layers=2, batch_size=8, epochs=10, learning_rate=0.001, weight_decay=1e-4, clip_value=5):
    # Load data and initialize dataset
    print('lets start')
    save_path_train = '/scratch-shared/ivanmiert/overview/train_testit_top_down_2.pt'
    save_path_val = '/scratch-shared/ivanmiert/overview/validation_testit_top_down_2.pt' 
    save_path_test = '/scratch-shared/ivanmiert/overview/test_set_testit_top_down_2.pt'
    save_path_dataset = '/scratch-shared/ivanmiert/overview/original_dataset_testit_top_down_2.pt'
    print(save_path_train)
    print(save_path_val)
    print(save_path_test)
    print(save_path_dataset)

    data = load_data_from_folder(folder_path, label_csv_path)
    dataset = ClipDataset(data, label_columns=['label', 'body_part_label', 'duel_label', 'pass_label', 'cross_accurate', 'cross_direction', 'cross_flank', 'pass_accurate', 'pass_direction', 'pass_distance', 'pass_progressive', 'pass_through', 'shot_on_target', 'shot_goal'])
    print('finished loading')
    torch.save(dataset, save_path_dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    torch.save(train_dataset, save_path_train)
    torch.save(val_dataset, save_path_val)
    torch.save(test_dataset, save_path_test)
    print(f'length train_dataset: {len(train_dataset)}')
    print(f'length val_dataset: {len(val_dataset)}')
    print(f'length test_dataset: {len(test_dataset)}')
