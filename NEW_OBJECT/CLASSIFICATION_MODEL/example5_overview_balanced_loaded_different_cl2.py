import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import time
import functools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

"""
This file contains the code on the creation, training, validating, evaluating of the classification models. 

"""



# Function to read CSV files and load data
def load_data_from_folder(folder_path, label_csv_path):
    label_df = pd.read_csv(label_csv_path)
    label_dict = dict(zip(label_df['clip_id'].astype(str), label_df['label']))

    # Define the columns to keep
    selected_columns = [
        'x_coordinate_pitch', 'y_coordinate_pitch', 'x_coordinate_pixel', 'y_coordinate_pixel',
        'x_min_coordinate_pixel', 'y_min_coordinate_pixel', 'x_max_coordinate_pixel', 'y_max_coordinate_pixel',
        'distance_to_field_keypoint_2', 'distance_to_field_keypoint_3', 'distance_to_field_keypoint_8',
        'distance_to_field_keypoint_9', 'distance_to_field_keypoint_12', 'distance_to_field_keypoint_13',
        'distance_to_field_keypoint_16', 'distance_to_field_keypoint_17', 'distance_to_field_keypoint_26',
        'distance_to_field_keypoint_27', 'distance_to_field_keypoint_28', 'distance_to_field_keypoint_29',
        'distance_to_field_keypoint_42', 'team_Team_1', 'team_Team_2', 'team_Goalkeeper_Team_1',
        'team_Goalkeeper_Team_2', 'team_Referee', 'player_object', 'ball_object'
    ]
    
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            #print(filename)
            clip_id = filename.replace('.csv', '')
            if clip_id not in label_dict or label_dict[clip_id] not in [0, 1]:  # Filter labels
                continue

            label = label_dict[clip_id]
            try:
                df = pd.read_csv(os.path.join(folder_path, filename))
            except pd.errors.EmptyDataError:
                print(f"Empty or malformed CSV file: {filename}")
                continue
            #df = pd.read_csv(os.path.join(folder_path, filename))

            # Fill missing values and select only the necessary columns
            df.fillna(0, inplace=True)
            df = df[['frame_number', 'ID'] + selected_columns]

            # Process data frame by frame
            frames = df['frame_number'].unique()
            frame_data = []
            for frame in frames:
                frame_df = df[df['frame_number'] == frame]
                # Only keep features from selected columns
                features = frame_df.drop(columns=['frame_number', 'ID']).values
                frame_data.append({'objects': frame_df['ID'].values, 'features': features})
            
            data.append({'frame_data': frame_data, 'label': label})

    return data

# Dataset class
# class ClipDataset(Dataset):
#     def __init__(self, data):
#         self.data = data
#         self.scaler = StandardScaler()
#         all_features = np.concatenate([frame['features'] for clip in data for frame in clip['frame_data']], axis=0)
#         self.scaler.fit(all_features)
#         self.label_encoder = LabelEncoder()
#         self.label_encoder.fit([clip['label'] for clip in data])
        
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         clip = self.data[idx]
#         for frame in clip['frame_data']:
#             frame['features'] = self.scaler.transform(frame['features'])
#         label = torch.tensor(self.label_encoder.transform([clip['label']])[0], dtype=torch.long)
#         return {'frame_data': clip['frame_data'], 'label': label}
    
#     def get_labels(self):
#         return self.labels
    
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
        
        return {'frame_data': clip['frame_data'], 'labels': labels}
    
    def get_labels(self, label_col):
        return [self.label_encoders[label_col].transform([clip[label_col]])[0] for clip in self.data]

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    

def collate_fn(batch, relevant_column):
    frame_data_list = [item['frame_data'] for item in batch]
    labels = torch.tensor([item['labels'][relevant_column] for item in batch], dtype=torch.long)
    clip_ids = [item['clip_id'] for item in batch] 
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
    
    return padded_sequences, labels, masks, clip_ids

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, lstm_output, mask):
        attention_scores = self.attention(lstm_output).squeeze(-1)
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(attention_scores, dim=1)
        return torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        attention_scores = self.attention(x).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        return torch.sum(x * attention_weights.unsqueeze(-1), dim=1)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout Layer
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.attention_pooling = AttentionPooling(hidden_size)
        
    def forward(self, x, mask):
        batch_size, max_frames, max_objects, feature_dim = x.size()
        print(f"x size before reshape: {x.size()}")
        print(f"batch_size: {batch_size}, max_frames: {max_frames}, max_objects: {max_objects}, feature_dim: {feature_dim}")
        x = x.view(batch_size * max_objects, max_frames, feature_dim)
        mask = mask.view(batch_size * max_objects, max_frames)
        lengths = mask.sum(dim=1)
        nonzero_indices = lengths.nonzero(as_tuple=True)[0]
        x_nonzero = x[nonzero_indices]
        mask_nonzero = mask[nonzero_indices]
        packed_input = nn.utils.rnn.pack_padded_sequence(x_nonzero, lengths[nonzero_indices].cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        aligned_mask = mask_nonzero[:, :output.size(1)]
        attended_output = self.attention(output, aligned_mask)
        out = self.dropout(torch.relu(self.fc1(attended_output)))  # Apply dropout
        final_output = torch.zeros(batch_size * max_objects, out.size(-1)).to(x.device)
        final_output[nonzero_indices] = out
        final_output = final_output.view(batch_size, max_objects, -1)
        pooled_output = self.attention_pooling(final_output)
        # Pass through the final fully connected layer (fc2) to get the logits for each class
        logits = self.fc2(pooled_output)
        return logits

def split_dataset(dataset, val_split=0.2, test_split=0.1):
    test_size = int(len(dataset)* test_split)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size - test_size
    return random_split(dataset, [train_size, val_size, test_size])

def load_data_for_model(dataset, model_label_column, relevant_labels):
    # print(model_label_column)
    # print(relevant_labels)
    # filtered_data = {key: value for key, value in dataset.items() if value[model_label_column] in relevant_labels}
    filtered_data = []
    print(f"relevant labels: {relevant_labels}")
    print(f"model column: {model_label_column}")
    # Iterate over the dataset (assuming each item in dataset is a dict or tuple)
    for item in dataset:
        print(item['labels'])
        # Access the label from the 'labels' part of the item dictionary
        label_value = item['labels'][model_label_column].item()  # Convert tensor to a Python scalar
        print('hier item label value:')
        print(label_value)
        # Check if the label value is in relevant_labels
        if label_value in relevant_labels:
            print('yes in')
            filtered_data.append(item)
    
    return filtered_data


def train_model(train_path, val_path, test_path, original_dataset_path, model_save_path, num_layers, num_features, relevant_column, relevant_labels, classification_sort, hidden_size=128, batch_size=16, epochs=100, learning_rate=0.001, weight_decay=1e-4, clip_value=5, dropout_rate=0.3):
    print(f"hidden: {hidden_size}, layers: {num_layers}, dropout: {dropout_rate}, batch_size: {batch_size}, learning rate: {learning_rate}")
    
    # Load pre-saved datasets
    time_start_loading = time.time()
    print('Loading datasets...')
    train_dataset = torch.load(train_path)
    print('training loaded')
    val_dataset = torch.load(val_path)
    print('validation loaded')
    test_dataset = torch.load(test_path)
    print('test loaded')

    # Load the original dataset to access label encoders
    original_dataset = torch.load(original_dataset_path)
    print('original loaded')
    time_end_loading = time.time()
    print(f"time loading: {time_end_loading - time_start_loading}")

    time_start_loading2 = time.time()
    train_set = load_data_for_model(train_dataset, relevant_column, relevant_labels)
    val_set = load_data_for_model(val_dataset, relevant_column, relevant_labels)
    test_set = load_data_for_model(test_dataset, relevant_column, relevant_labels)
    time_end_loading2 = time.time()
    print(f"time loading 2: {time_end_loading2 - time_start_loading2}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare label encoders
    train_labels = [item['labels'][relevant_column].item() for item in train_set] #HIER LABEL
    val_labels = [item['labels'][relevant_column].item() for item in val_set] #HIER LABEL
    test_labels = [item['labels'][relevant_column].item() for item in test_set] #HIER LABEL
    print(f"length labels; training: {len(train_labels)}, validation: {len(val_labels)}, test: {len(test_labels)}")
    
    label_encoder = original_dataset.label_encoders[relevant_column] #HIER LABEL
    encoded_train_labels = label_encoder.fit_transform(train_labels)
    encoded_val_labels = label_encoder.transform(val_labels)
    encoded_test_labels = label_encoder.transform(test_labels)

    # Class counts and sampling
    class_counts = np.bincount(encoded_train_labels)
    sample_weights = np.array([1.0 / class_counts[label] for label in encoded_train_labels])
    train_sampler = WeightedRandomSampler(sample_weights, len(train_set))
    custom_collate_fn = functools.partial(collate_fn, relevant_column=relevant_column)
    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=custom_collate_fn)
    num_classes = len(class_counts)

    # Model with dropout
    model = LSTMClassifier(num_features, hidden_size, num_layers, num_classes, dropout_rate=dropout_rate).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = -1
    base_path = os.path.splitext(model_save_path)[0]
    model_save_path = f"{base_path}_BOTTOM_UP_{relevant_column}_{hidden_size}_{batch_size}_{learning_rate}_{dropout_rate}_{num_layers}.pt"

    begin_time = time.time()

    # Training loop with validation
    for epoch in range(epochs):
        start_time_epoch = time.time()
        model.train()
        train_loss = 0
        label_counts = {i: 0 for i in range(num_classes)}
        
        for inputs, labels, mask, clip_ids in train_loader:
            inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, mask)
            if classification_sort != 'primary':
                labels = labels-1
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            train_loss += loss.item()
            for label in labels.cpu().numpy():
                label_counts[label] += 1

        end_time_epoch = time.time()
        train_loss /= len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Time: {end_time_epoch - start_time_epoch:.2f}s')
        
        # Validation loop
        model.eval()
        val_loss = 0
        correct, total = 0, 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels, mask, clip_ids in val_loader:
                inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
                if classification_sort != 'primary':
                    labels = labels -1
                outputs = model(inputs, mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader)
        accuracy = correct / total * 100
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved at epoch {best_epoch} with validation loss {best_val_loss:.4f} and accuracy {best_val_acc:.2f}%")
    end_time = time.time()
    print(f"TOTAL TRAINING TIME IS {end_time - begin_time}")
    print(f"Best model was at epoch {best_epoch} with validation accuracy {best_val_acc:.2f}%")
    model.load_state_dict(torch.load(model_save_path))
    return model, test_set

# Evaluate the model
def evaluate_model(model, test_set, num_classes, dataframe_save_path, relevant_column, hidden_size, batch_size_train, learning_rate, dropout_rate, number_of_layers, classification_sort, batch_size=32):
    # print('testset:')
    # print(test_set)
    custom_collate_fn = functools.partial(collate_fn, relevant_column=relevant_column)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=custom_collate_fn)
    # print('test_loader:')
    # print(test_loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    correct, total = 0, 0
    all_labels = []
    all_predictions = []
    all_clip_ids = []

    with torch.no_grad():
        for inputs, labels, mask, clip_ids in test_loader:
            # print(inputs.shape)
            # print('hier labels:')
            # print(labels)
            # print(clip_ids)
            inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
            outputs = model(inputs, mask)
            if classification_sort != 'primary':
                labels = labels-1
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store the labels and predictions
            all_clip_ids.extend(clip_ids)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate accuracy
    accuracy = correct / total * 100
    print(f'Accuracy: {accuracy:.2f}%')
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f"precision {precision}, recall {recall}, f1 {f1}")
    # Combine all labels, predictions, and clip_ids into a DataFrame
    evaluation_df = pd.DataFrame({
        'clip_id': all_clip_ids,
        'true_label': all_labels,
        'predicted_label': all_predictions
    })
    base_path = os.path.splitext(dataframe_save_path)[0]
    eval_df_path = f"{base_path}_BOTTOM_UP_{hidden_size}_{batch_size_train}_{learning_rate}_{dropout_rate}_{number_of_layers}.csv"
    
    # Save the DataFrame as a CSV
    try:
        evaluation_df.to_csv(eval_df_path, index=False)
        print(f"Evaluation results successfully saved to {eval_df_path}")
    except Exception as e:
        print(f"Error saving evaluation results: {e}")

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=[f'Class {i}' for i in range(num_classes)],
                                  columns=[f'Class {i}' for i in range(num_classes)])

    print("Confusion Matrix:\n", conf_matrix_df)
    conf_base_path = base_path.replace('.csv', '_conf_matrix.csv')
    conf_base_path_splitted = os.path.splitext(conf_base_path)[0]
    conf_df_path = f"{conf_base_path_splitted}_BOTTOM_UP_CONF_{hidden_size}_{batch_size_train}_{learning_rate}_{dropout_rate}_{number_of_layers}.csv"
    # Save confusion matrix
    try:
        conf_matrix_df.to_csv(conf_df_path)
        print(f"Confusion matrix successfully saved to {conf_df_path}")
    except Exception as e:
        print(f"Error saving confusion matrix: {e}")
