import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from scipy.stats import mode

# Parameters
base_folder = "D:\\Footage\\Classifier\\train"  # Replace with your base folder path
batch_size = 32
num_clusters = 11
intra_cluster_penalty = 0.3
inter_cluster_penalty = 1.0

def extract_features_and_create_penalty_matrix(base_folder, batch_size, num_clusters, intra_cluster_penalty, inter_cluster_penalty):
    # Step 1: Load Images and Extract Features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define a transform to preprocess the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the dataset
    dataset = datasets.ImageFolder(base_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load a pre-trained model for feature extraction
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove the classification layer
    model = model.to(device)
    model.eval()

    # Extract features
    features = []
    labels = []

    with torch.no_grad():
        for inputs, label_batch in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)
            features.append(outputs.cpu().numpy())
            labels.extend(label_batch.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.array(labels)

    # Step 2: Cluster the classes based on their features
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    # Step 3: Map the clusters back to the classes
    class_cluster_mapping = {}
    for class_idx in range(len(dataset.classes)):
        class_features = features[labels == class_idx]
        class_cluster_labels = kmeans.predict(class_features)
        # Handle the mode result safely
        class_cluster_label = mode(class_cluster_labels).mode
        if class_cluster_label.size > 0:
            print(class_cluster_label)
            class_cluster_label = class_cluster_label
            print(class_cluster_label)
        else:
            print('ello')
            print(class_cluster_label)
            class_cluster_label = class_cluster_labels[0]  # fallback in case of an empty mode
            print(class_cluster_label)
        class_cluster_mapping[class_idx] = class_cluster_label

        # Debug: Print the cluster assignments
    print("Class to Cluster Mapping:")
    for class_idx, cluster_label in class_cluster_mapping.items():
        print(f"Class {class_idx} ({dataset.classes[class_idx]}): Cluster {cluster_label}")

    # Step 4: Initialize the penalty matrix
    num_classes = len(dataset.classes)
    penalty_matrix = np.zeros((num_classes, num_classes))

    # Step 5: Assign penalties based on cluster membership
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                if class_cluster_mapping[i] == class_cluster_mapping[j]:
                    penalty_matrix[i, j] = intra_cluster_penalty
                else:
                    penalty_matrix[i, j] = inter_cluster_penalty

    # Step 6: Save the penalty matrix
    output_dir = 'D:\\GITHUB\\AdvancedDataPreProcessingMethods\\NEW_OBJECT\\files_folder'  # Replace with your desired output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save as .npy file
    npy_file_path = os.path.join(output_dir, 'penalty_matrix.npy')
    np.save(npy_file_path, penalty_matrix)

    # Save as .csv file
    csv_file_path = os.path.join(output_dir, 'penalty_matrix.csv')
    np.savetxt(csv_file_path, penalty_matrix, delimiter=',')

    # Print confirmation
    print(f"Penalty matrix saved to {npy_file_path} and {csv_file_path}")

    # Print the penalty matrix
    print("Penalty Matrix:")
    print(penalty_matrix)

if __name__ == '__main__':
    extract_features_and_create_penalty_matrix(base_folder, batch_size, num_clusters, intra_cluster_penalty, inter_cluster_penalty)