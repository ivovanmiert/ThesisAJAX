import matplotlib.pyplot as plt
import numpy as np

# Data
classification_types = [
    'Primary', 'Shot Body Part', 'Duel', 'Normal Pass / Cross', 
    'Accuracy Cross', 'Direction Cross', 'Flank Cross', 
    'Accuracy Pass', 'Direction Pass', 'Distance Pass', 
    'Progressive Pass', 'Through Pass', 'On Target Shot'
]

base_accuracy = [
    33.19, 39.22, 53.77, 91.19, 
    67.39, 76.09, 50.00, 
    90.27, 54.87, 12.39, 
    82.30, 71.68, 51.69
]

top_down_accuracy = [
    38.05, 39.02, 41.57, 90.80, 
    50.88, 38.60, 75.44, 
    82.08, 42.45, 83.02, 
    52.83, 55.66, 53.66
]

bottom_up_accuracy = [
    33.63, 25.00, 47.57, 94.67, 
    55.07, 72.46, 40.58, 
    86.00, 40.00, 87.00, 
    71.00, 61.00, 47.73
]

base_f1_score = [
    0.3175, 0.3647, 0.4472, 0.9134, 
    0.5426, 0.7681, 0.4043, 
    0.8710, 0.4442, 0.0273, 
    0.7431, 0.5986, 0.4692
]

top_down_f1_score = [
    0.3281, 0.2369, 0.3272, 0.9082, 
    0.5057, 0.4865, 0.7665, 
    0.7626, 0.4039, 0.8271, 
    0.5667, 0.5592, 0.4689
]

bottom_up_f1_score = [
    0.3198, 0.2436, 0.4803, 0.9469, 
    0.3912, 0.7184, 0.4213, 
    0.8045, 0.3529, 0.8795, 
    0.6712, 0.5682, 0.4738
]

# Bar width
bar_width = 0.25

# Bar positions
r1 = np.arange(len(classification_types))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Accuracy Bar Plot
plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
plt.bar(r1, base_accuracy, color=colors[0], width=bar_width, edgecolor='grey', label='Base Accuracy')
plt.bar(r2, top_down_accuracy, color=colors[1], width=bar_width, edgecolor='grey', label='Top-Down Accuracy')
plt.bar(r3, bottom_up_accuracy, color=colors[2], width=bar_width, edgecolor='grey', label='Bottom-Up Accuracy')

# Labels and title for accuracy plot
plt.xlabel('Classification Type', fontweight='bold', fontsize=12)
plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
plt.title('Accuracy by Classification Type', fontweight='bold', fontsize=14)
plt.xticks([r + bar_width for r in range(len(classification_types))], classification_types, rotation=45, ha='right', fontsize=10)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the accuracy plot
plt.tight_layout()
plt.savefig('/home/ivanmiert/overview/run_classification/dataframes_evaluation/output/accuracy_by_classification_type.png', dpi=300)
plt.close()  # Close the figure to avoid displaying it

# F1 Score Bar Plot
plt.figure(figsize=(10, 6))
plt.bar(r1, base_f1_score, color=colors[0], width=bar_width, edgecolor='grey', label='Base F1 Score')
plt.bar(r2, top_down_f1_score, color=colors[1], width=bar_width, edgecolor='grey', label='Top-Down F1 Score')
plt.bar(r3, bottom_up_f1_score, color=colors[2], width=bar_width, edgecolor='grey', label='Bottom-Up F1 Score')

# Labels and title for F1 score plot
plt.xlabel('Classification Type', fontweight='bold', fontsize=12)
plt.ylabel('F1 Score', fontweight='bold', fontsize=12)
plt.title('F1 Score by Classification Type', fontweight='bold', fontsize=14)
plt.xticks([r + bar_width for r in range(len(classification_types))], classification_types, rotation=45, ha='right', fontsize=10)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the F1 score plot
plt.tight_layout()
plt.savefig('/home/ivanmiert/overview/run_classification/dataframes_evaluation/output/f1_score_by_classification_type.png', dpi=300)
plt.close()  # Close the figure to avoid displaying it