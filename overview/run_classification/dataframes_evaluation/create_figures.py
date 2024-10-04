import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

def load_data(file_path):
    return pd.read_csv(file_path)

def plot_precision_recall_f1(true_labels, pred_labels, class_names, save_path):
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None)
    
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

    bar_width = 0.2
    x = np.arange(len(class_names))

    plt.figure(figsize=(12, 6))
    plt.bar(x - bar_width, metrics_df['Precision'], width=bar_width, label='Precision', color='blue')
    plt.bar(x, metrics_df['Recall'], width=bar_width, label='Recall', color='orange')
    plt.bar(x + bar_width, metrics_df['F1 Score'], width=bar_width, label='F1 Score', color='green')

    plt.xlabel('Classes', fontsize=14)
    plt.ylabel('Scores', fontsize=14)
    plt.title('Per-Class Precision, Recall, and F1 Scores', fontsize=16)
    plt.xticks(x, metrics_df['Class'], fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix_with_percentages(true_labels, pred_labels, class_names, save_path):
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  

    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16})  
    
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    
    row_percentages = np.round(np.diag(cm_normalized), 2)  
    
    for i, percentage in enumerate(row_percentages):
        plt.text(i + 0.5, len(class_names) + 0.5, f'{percentage}%', 
                 ha='center', va='center', fontsize=14, color='black') 

    col_percentages = np.round(np.diag(cm_normalized.T), 2)
    for i, percentage in enumerate(col_percentages):
        plt.text(len(class_names) + 0.5, i + 0.5, f'{percentage}%', 
                 ha='center', va='center', fontsize=14, color='black') 

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_error_matrix(true_labels, pred_labels, class_names, save_path):
    cm = confusion_matrix(true_labels, pred_labels)
    
    error_matrix = cm.copy()
    np.fill_diagonal(error_matrix, 0)  
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(error_matrix, annot=True, fmt=".0f", cmap='Reds', cbar=False, 
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16}) 
    
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.title('Error Matrix (Misclassifications Only)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main(file_path, output_dir):
    data = load_data(file_path)
    
    true_labels = data['true_label']
    predicted_labels = data['predicted_label']
    
    class_labels = ['Shot', 'Pass', 'Duel', 'Interception', 'Touch']
    
    plot_confusion_matrix_with_percentages(true_labels, predicted_labels, class_labels, f'{output_dir}/confusion_matrix_with_percentages.png')
    plot_precision_recall_f1(true_labels, predicted_labels, class_labels, f'{output_dir}/precision_recall_f1.png')
    plot_error_matrix(true_labels, predicted_labels, class_labels, f'{output_dir}/error_matrix.png')

file_path = '/home/ivanmiert/overview/run_classification/dataframes_evaluation/basicprimary13064640.0010.34_BOTTOM_UP_64_64_0.001_0.3_4.csv'
output_dir = '/home/ivanmiert/overview/run_classification/dataframes_evaluation/output' 
main(file_path, output_dir)