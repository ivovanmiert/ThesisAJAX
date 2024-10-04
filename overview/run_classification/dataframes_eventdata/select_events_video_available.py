import os
import pandas as pd


#The first part of this file takes the 10.000 events that were selected, and checks which events are in the games from which the video footage is there. 
#From the 10.000 selected events, 8069 events are in the videos that are available. 

# Load your dataframe (replace with the correct path to your CSV or dataframe)
df = pd.read_csv('/home/ivanmiert/overview/FRAMES_EXTRACTION/dataframes_eventdata/all_selected.csv')

# Folder where the videos are stored
video_folder = '/scratch-shared/ivanmiert/All_matches_27aug'

# Get a list of all game_ids from the videos in the folder
videos = os.listdir(video_folder)
game_ids = [int(video.split('-')[0].strip('g')) for video in videos if video.endswith('.mp4')]
#print(game_ids)

#print(df.dtypes)
# Filter the dataframe to include only rows where 'match-id' is in game_ids
filtered_df = df[df['match_id'].isin(game_ids)]
filtered_df.to_csv('/home/ivanmiert/overview/FRAMES_EXTRACTION/dataframes_eventdata/video_available_selected.csv')


#This second part of this file takes from the 8069 events, and splits them up into smaller dataframes of 1000 actions per piece

# Shuffle the dataframe
shuffled_df = filtered_df.sample(frac=1).reset_index(drop=True)

# Define the output folder where the smaller DataFrames will be saved
output_folder = '/home/ivanmiert/overview/FRAMES_EXTRACTION/dataframes_eventdata/dataframes_chunks_1000'
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Split the dataframe into chunks of 1000 rows each
chunk_size = 1000
for i in range(0, len(shuffled_df), chunk_size):
    chunk_df = shuffled_df.iloc[i:i + chunk_size]
    
    # Save each chunk as a CSV file
    chunk_filename = f'{output_folder}/chunk_{i//chunk_size + 1}.csv'
    chunk_df.to_csv(chunk_filename, index=False)

print(f"Saved {len(shuffled_df) // chunk_size} chunks of 1000 rows each.")