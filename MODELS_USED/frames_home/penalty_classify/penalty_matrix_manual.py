import numpy as np
import os
import pandas as pd
# List of all classes (from your description)
classes = [
    "ac_milan_1", "ac_milan_2", "ac_milan_3", "ac_milan_4", "ac_milan_Keeper1", 
    "ac_milan_Keeper2", "ac_milan_Keeper3", "ac_milan_Keeper4", "atalanta_1", 
    "atalanta_2", "atalanta_3", "atalanta_Keeper1", "atalanta_Keeper2", 
    "atalanta_Keeper3", "bologna_1", "bologna_2", "bologna_3", "bologna_Keeper2", 
    "bologna_Keeper3", "bologna_Keeper4", "cagliari_1", "cagliari_2", 
    "cagliari_3", "cagliari_Keeper1", "cagliari_Keeper2", "empoli_1", "empoli_2", 
    "empoli_3", "empoli_Keeper1", "empoli_Keeper2", "empoli_Keeper3", "empoli_Keeper5", "empoli_Keeper6", "fiorentina_1", 
    "fiorentina_2", "fiorentina_3", "fiorentina_Keeper1", "fiorentina_Keeper2", 
    "fiorentina_Keeper3", "frosinone_1", "frosinone_2", "frosinone_3", 
    "frosinone_Keeper1", "frosinone_Keeper3", "frosinone_Keeper4", "genoa_1", 
    "genoa_2", "genoa_3", "genoa_Keeper1", "genoa_Keeper2", "genoa_Keeper3", 
    "inter_milan_1", "inter_milan_2", "inter_milan_3", "inter_milan_Keeper1", 
    "inter_milan_Keeper2", "inter_milan_Keeper3", "inter_milan_Keeper4", 
    "juventus_1", "juventus_2", "juventus_3", "juventus_Keeper1", 
    "juventus_Keeper2", "juventus_Keeper3", "lazio_1", "lazio_2", "lazio_3", 
    "lazio_Keeper1", "lazio_Keeper2", "lazio_Keeper3", "lecce_1", 
    "lecce_2", "lecce_3", "lecce_Keeper1", "lecce_Keeper2", "lecce_Keeper3", "lecce_Keeper4",
    "monza_1", "monza_2", "monza_3", "monza_Keeper1", 
    "monza_Keeper2", "monza_Keeper3", "monza_Keeper4", "monza_Keeper5", 
    "napoli_1", "napoli_2", "napoli_3", "napoli_Keeper2", 
    "napoli_Keeper3", "napoli_Keeper4", "referee_1", "referee_2", "referee_3", 
    "referee_4", "roma_1", "roma_2", "roma_3", "roma_Keeper1", "roma_Keeper2", 
    "roma_Keeper3", "salernitana_1", "salernitana_2", "salernitana_3", 
    "salernitana_Keeper2", "salernitana_Keeper3", "salernitana_Keeper4", 
    "sassuolo_1", "sassuolo_2", "sassuolo_3", "sassuolo_Keeper1", 
    "sassuolo_Keeper2", "sassuolo_Keeper3", "sassuolo_Keeper4", "torino_1", 
    "torino_2", "torino_Keeper1", "torino_Keeper2", "torino_Keeper3", "udinese_1", 
    "udinese_2", "udinese_3", "udinese_Keeper1", "udinese_Keeper2", 
    "udinese_Keeper3", "verona_1", "verona_2", "verona_3", "verona_4", 
    "verona_Keeper1", "verona_Keeper2", "verona_Keeper3"
]
print(len(classes))
test = ['hello']
print(len(test))

clusters = [
    #ROOD
    ["ac_milan_1", "atalanta_3", "lecce_3", "monza_1", "monza_Keeper4", "roma_1", "salernitana_1", "torino_1"], #8
    #WIT
    ["ac_milan_2", "ac_milan_Keeper4", "atalanta_2", "bologna_2", "bologna_3", "cagliari_2", "empoli_2", "empoli_Keeper3", "fiorentina_2", "frosinone_2", "genoa_2", "inter_milan_2", "juventus_1",  "juventus_2", "lazio_3", "lecce_2", "monza_2", "monza_Keeper3", "napoli_2", "roma_2", "salernitana_2",  "sassuolo_2", "sassuolo_Keeper3", "torino_2",  "udinese_1", "verona_3" ], #26
    #BLACK
    ["ac_milan_4", "ac_milan_Keeper3", "bologna_Keeper4", "cagliari_3", "empoli_3", "empoli_Keeper5", "inter_milan_Keeper3", "juventus_3", "lazio_2", "lazio_Keeper1", "lecce_Keeper1", "monza_3", "monza_Keeper2", "napoli_3", "napoli_Keeper4", "referee_1", "roma_3", "salernitana_3", "torino_Keeper3", "udinese_3", "verona_4", "verona_Keeper2"], #22
    #GREEN
    ["ac_milan_Keeper1", "atalanta_Keeper3", "bologna_Keeper3", "cagliari_Keeper2", "fiorentina_Keeper2", "frosinone_Keeper1", "inter_milan_Keeper2", "juventus_Keeper3", "roma_Keeper2", "sassuolo_1", "torino_Keeper2", "udinese_Keeper1", "verona_Keeper1"], #13
    #ORANGE
    ["ac_milan_Keeper2", "atalanta_Keeper2", "bologna_Keeper2", "empoli_Keeper2", "fiorentina_Keeper1", "frosinone_Keeper4", "inter_milan_3", "inter_milan_Keeper1", "juventus_Keeper2", "lazio_Keeper2", "napoli_Keeper3", "referee_3", "sassuolo_Keeper2", "torino_Keeper1"], #14
    #BLUE
    ["atalanta_1", "cagliari_Keeper1", "empoli_1", "fiorentina_3", "fiorentina_Keeper3", "genoa_Keeper1", "inter_milan_1", "inter_milan_Keeper4", "juventus_Keeper1", "lazio_1", "lecce_Keeper2", "monza_Keeper5", "napoli_1", "referee_4", "roma_Keeper3", "salernitana_Keeper3", "sassuolo_Keeper1", "verona_1" ], #18
    #PURPLE
    ["ac_milan_3", "fiorentina_1", "frosinone_Keeper3"], #3
    #PINK
    ["salernitana_Keeper4", "udinese_2", "udinese_Keeper3", "verona_Keeper3" ], #4
    #YELLOW
    ["atalanta_Keeper1", "empoli_Keeper1", "frosinone_1", "genoa_3", "genoa_Keeper2", "genoa_Keeper3", "lazio_Keeper3", "lecce_1", "lecce_Keeper3", "monza_Keeper1", "napoli_Keeper2",  "referee_2", "roma_Keeper1", "salernitana_Keeper2", "sassuolo_3", "sassuolo_Keeper4", "verona_2"], #17
    #RED/BLUE
    ["bologna_1", "cagliari_1", "genoa_1"], #3
    #GRAY
    ["empoli_Keeper6", "frosinone_3", "lecce_Keeper4", "udinese_Keeper2"], #4
]

# # Define the clusters (each list contains the classes in that cluster)
# clusters = [
#     ["ac_milan_1", "lecce_3", "atalanta_3", "monza_1", "monza_Keeper4", "roma_1", "salernitana_1", "sassuolo_Keeper2", "torino_1", "sassuolo_2", "sassuolo_Keeper3", "torino_2", "udinese_1", "verona_3"],
#     ["ac_milan_2", "ac_milan_Keeper4", "atalanta_2", "bologna_2", "bologna_3", "cagliari_2", "empoli_2", "empoli_Keeper3", "fiorentina_2", "frosinone_2", "genoa_2", "inter_milan_2", "juventus_2", "lazio_3", "lecce_2", "monza_2", "monza_Keeper3", "napoli_2", "referee_1", "roma_2", "salernitana_2"],
#     ["ac_milan_4", "ac_milan_Keeper3", "bologna_Keeper4", "cagliari_3", "empoli_3", "empoli_Keeper5", "inter_milan_Keeper3", "juventus_1", "juventus_3", "lazio_Keeper1", "lecce_Keeper1", "monza_3", "monza_Keeper2", "napoli_3", "napoli_Keeper4", "roma_3", "salernitana_3", "torino_Keeper3", "udinese_3", "verona_4", "verona_Keeper2"],
#     ["ac_milan_Keeper1", "atalanta_Keeper3", "cagliari_Keeper2", "fiorentina_Keeper2", "frosinone_Keeper1", "inter_milan_Keeper2", "roma_Keeper2", "sassuolo_1", "torino_Keeper2", "udinese_Keeper1", "verona_Keeper1"],
#     ["ac_milan_Keeper2", "atalanta_Keeper2", "bologna_Keeper2", "empoli_Keeper2", "fiorentina_Keeper1", "frosinone_Keeper4", "inter_milan_3", "inter_milan_Keeper1", "juventus_Keeper2", "lazio_Keeper2", "lecce_Keeper2", "monza_Keeper5", "napoli_1", "referee_4"],
#     ["atalanta_1", "bologna_Keeper3", "cagliari_Keeper1", "empoli_1", "fiorentina_3", "fiorentina_Keeper3", "genoa_Keeper1", "inter_milan_1", "inter_milan_Keeper4", "juventus_Keeper1", "lazio_1", "lazio_2"],
#     ["ac_milan_3", "fiorentina_1", "frosinone_Keeper3"],
#     ["salernitana_Keeper4", "udinese_2", "udinese_Keeper3", "verona_Keeper3"],
#     ["atalanta_Keeper1", "empoli_Keeper1", "frosinone_1", "genoa_3", "genoa_Keeper2", "genoa_Keeper3", "juventus_Keeper3", "lazio_Keeper3", "lecce_1", "lecce_Keeper3", "monza_Keeper1", "napoli_Keeper2", "referee_2", "roma_Keeper1", "salernitana_Keeper2"],
#     ["bologna_1", "cagliari_1", "genoa_1"],
#     ["empoli_Keeper6", "frosinone_3", "lecceKeeper4", "udinese_Keeper2"]
# ]

# # Function to create the penalty matrix
# def create_penalty_matrix(classes, clusters, intra_cluster_penalty, inter_cluster_penalty):
#     num_classes = len(classes)
#     penalty_matrix = np.ones((num_classes, num_classes)) * inter_cluster_penalty

#     # Create a dictionary to map classes to their clusters
#     class_to_cluster = {}
#     i = 0
#     for cluster_idx, cluster in enumerate(clusters):
#         for class_name in cluster:
#             i += 1
#             print(i)
#             class_to_cluster[class_name] = cluster_idx

#     #print(i)
#     #print(len(class_to_cluster))
#     #print(class_to_cluster)
#     # Assign penalties
#     for i, class1 in enumerate(classes):
#         for j, class2 in enumerate(classes):
#             if i == j:
#                 penalty_matrix[i, j] = 0  # Zero penalty for the same class
#             elif class_to_cluster.get(class1) == class_to_cluster.get(class2):
#                 penalty_matrix[i, j] = intra_cluster_penalty

#     return penalty_matrix

def create_penalty_matrix(classes, clusters, intra_cluster_penalty, inter_cluster_penalty):
    num_classes = len(classes)
    print(num_classes)
    penalty_matrix = np.ones((num_classes, num_classes)) * inter_cluster_penalty
    
    # Create a DataFrame for the penalty matrix
    penalty_df = pd.DataFrame(penalty_matrix, index=classes, columns=classes)

    # Create a dictionary to map classes to their clusters
    class_to_cluster = {}
    for cluster_idx, cluster in enumerate(clusters):
        for class_name in cluster:
            class_to_cluster[class_name] = cluster_idx

    # Convert the dictionary to a Series for easy access
    class_to_cluster_series = pd.Series(class_to_cluster)

    # Assign penalties
    for class1 in classes:
        for class2 in classes:
            if class1 == class2:
                penalty_df.at[class1, class2] = 0  # Zero penalty for the same class
            elif class_to_cluster_series.get(class1) == class_to_cluster_series.get(class2):
                penalty_df.at[class1, class2] = intra_cluster_penalty
    print(penalty_df.shape)
    return penalty_df

# Define penalties
intra_cluster_penalty = 0.3
inter_cluster_penalty = 1
#pd.options.display.max_rows = 150
#pd.options.display.max_columns = 150
# Create the penalty matrix
penalty_matrix = create_penalty_matrix(classes, clusters, intra_cluster_penalty, inter_cluster_penalty)
#print(penalty_matrix)
# Display the matrix
#print(penalty_matrix)

output_dir = '/home/ivanmiert/frames_home/penalty_classify'
# Save as .csv file
csv_file_path = os.path.join(output_dir, 'penalty_matrix_manual_03_corrected.csv')
penalty_matrix.to_csv(csv_file_path)

penalty_matrix = pd.read_csv('/home/ivanmiert/frames_home/penalty_classify/penalty_matrix_manual_03_corrected.csv', index_col=0)
print(penalty_matrix)