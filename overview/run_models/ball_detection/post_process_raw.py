import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import cv2
import os

def count_frames(frames_folder):
    # List all files in the folder
    files = os.listdir(frames_folder)
    
    # Filter out files that are not frames (based on naming convention, e.g., '.jpg' extension)
    frame_files = [f for f in files if f.endswith('.jpg')]  # Adjust if your frame format differs
    
    # Return the number of frame files
    return len(frame_files)

def process_csv_files(folder_path, frames_folder_base, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            # Skip if the CSV file already exists
            #print(file_path)
            data = pd.read_csv(file_path, 
                               names=['frame_number', 'class', 'x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'color'])

            # Extract event number and derive the frames folder path
            event_number = filename.split('.')[0].split('_')[-1]
            truncated_event_number = event_number[:10]
            frames_folder = os.path.join(frames_folder_base, f"event_{truncated_event_number}")
            number_of_frames = count_frames(frames_folder)

            # Output file path
            output_csv_filename = f"event_{truncated_event_number}.csv"
            output_csv_path = os.path.join(output_folder, output_csv_filename)
            if os.path.exists(output_csv_path):
                print(f'{output_csv_path} already exists, skipping...')
                continue

            # Convert relevant columns to numeric
            cols_to_convert = ['x_min', 'y_min', 'x_max', 'y_max', 'frame_number']
            data[cols_to_convert] = data[cols_to_convert].apply(pd.to_numeric, errors='coerce')

            # Compute the center of the bounding boxes
            data['x_center'] = (data['x_min'] + data['x_max']) / 2
            data['y_center'] = (data['y_min'] + data['y_max']) / 2

            # Sort the data by frame number
            data.sort_values(by='frame_number', inplace=True)

            # Compute moving averages
            window_size = 10
            data['x_moving_avg'] = data['x_center'].rolling(window=window_size).mean()
            data['y_moving_avg'] = data['y_center'].rolling(window=window_size).mean()

            # Calculate deviations from the moving average
            data['x_deviation'] = abs(data['x_center'] - data['x_moving_avg'])
            data['y_deviation'] = abs(data['y_center'] - data['y_moving_avg'])

            # Calculate deviations and IQR-based thresholds
            Q1_x, Q3_x = data['x_center'].quantile([0.25, 0.75])
            IQR_x = Q3_x - Q1_x
            deviation_threshold_x = 1.5 * IQR_x

            Q1_y, Q3_y = data['y_center'].quantile([0.25, 0.75])
            IQR_y = Q3_y - Q1_y
            
            # Define dynamic thresholds
            deviation_threshold_x = 1.5 * IQR_x  # 1.5 is a typical factor for IQR outlier detection
            deviation_threshold_y = 1.5 * IQR_y

            # Mark deviations and large frame gaps
            data['outlier_deviation'] = (data['x_deviation'] > deviation_threshold_x) | (data['y_deviation'] > deviation_threshold_y)
            data['frame_gap'] = data['frame_number'].diff().fillna(0)
            large_gap_threshold = 10
            data['outlier_gap'] = data['frame_gap'] > large_gap_threshold

        # Combine both criteria for outliers
        data['outlier'] = data['outlier_deviation'] #| data['outlier_gap']

        data.dropna(subset=['x_center', 'y_center'], inplace=True)
        # Use DBSCAN for spatial-temporal clustering
        # Prepare data for clustering (x, y, and frame_number)
        X = data[['x_center', 'y_center', 'frame_number']].values
        #print(X)
        if len(X) == 0:
            print(f"No data available for DBSCAN clustering in file {filename}.")
            continue  # Skip this file if no valid data
        
        dbscan = DBSCAN(eps=50, min_samples=3)  # eps and min_samples need to be adjusted for your data
        data['dbscan_label'] = dbscan.fit_predict(X)

        # Mark points that are considered noise (-1 label in DBSCAN) as outliers
        data['outlier_dbscan'] = data['dbscan_label'] == -1

        # Final outlier detection combining all methods
        data['final_outlier'] = data['outlier'] | data['outlier_dbscan']

        data = data[data['final_outlier']==False]

        # Create a helper column to count the occurrences of each frame_number
        data['frame_count'] = data.groupby('frame_number')['frame_number'].transform('count')

        # Function to interpolate between two values
        def interpolate(prev_val, next_val, prev_frame, next_frame, current_frame):
            return prev_val + ((next_val - prev_val) / (next_frame - prev_frame)) * (current_frame - prev_frame)

        # Store rows to keep
        rows_to_keep = pd.DataFrame()

        # Iterate through the rows with more than one detection for the same frame
        for frame_number, group in data[data['frame_count'] > 1].groupby('frame_number'):
            
            # Get the previous and next frames with only one detection
            prev_frame = data[(data['frame_number'] < frame_number) & (data['frame_count'] == 1)].tail(1)
            next_frame = data[(data['frame_number'] > frame_number) & (data['frame_count'] == 1)].head(1)
            
            # Initialize interpolated center
            interpolated_x, interpolated_y = None, None
            
            # If both previous and next frames exist
            if not prev_frame.empty and not next_frame.empty:
                # Get the center values for the previous and next frames
                prev_x_center = prev_frame['x_center'].values[0]
                prev_y_center = prev_frame['y_center'].values[0]
                next_x_center = next_frame['x_center'].values[0]
                next_y_center = next_frame['y_center'].values[0]
                
                prev_frame_number = prev_frame['frame_number'].values[0]
                next_frame_number = next_frame['frame_number'].values[0]
                
                # Interpolate x_center and y_center for the current frame
                interpolated_x = interpolate(prev_x_center, next_x_center, prev_frame_number, next_frame_number, frame_number)
                interpolated_y = interpolate(prev_y_center, next_y_center, prev_frame_number, next_frame_number, frame_number)
            
            # If only the previous frame exists
            elif not prev_frame.empty:
                interpolated_x = prev_frame['x_center'].values[0]
                interpolated_y = prev_frame['y_center'].values[0]
            
            # If only the next frame exists
            elif not next_frame.empty:
                interpolated_x = next_frame['x_center'].values[0]
                interpolated_y = next_frame['y_center'].values[0]
            
            # Only proceed if an interpolated point exists (i.e., there's at least one surrounding frame)
            if interpolated_x is not None and interpolated_y is not None:
                # Calculate the distance to the interpolated value for each detection in the current frame
                group['distance_to_interpolated'] = np.sqrt((group['x_center'] - interpolated_x) ** 2 +
                                                            (group['y_center'] - interpolated_y) ** 2)
                
                # Find the detection closest to the interpolated value or the nearest available frame
                closest_detection = group.loc[group['distance_to_interpolated'].idxmin()]
                
                # Append the closest detection to the rows to keep
                rows_to_keep = pd.concat([rows_to_keep, closest_detection.to_frame().T])

        # Remove the frames that have multiple detections and only keep the rows with single detections or the closest one
        data_cleaned = pd.concat([data[data['frame_count'] == 1], rows_to_keep])

        # Reset index if needed
        data_cleaned.reset_index(drop=True, inplace=True)

        data = data_cleaned
        # Re-index to make sure all frames are included
        #full_frame_range = pd.DataFrame({'frame_number': range(int(data['frame_number'].min()), int(data['frame_number'].max()) + 1)})
        full_frame_range = pd.DataFrame({'frame_number': range(0, number_of_frames)})
        data = pd.merge(full_frame_range, data, on='frame_number', how='left')
        # Convert relevant columns to numeric, errors='coerce' will turn invalid parsing into NaN
        cols_to_convert = ['x_min', 'y_min', 'x_max', 'y_max', 'x_center', 'y_center']
        data[cols_to_convert] = data[cols_to_convert].apply(pd.to_numeric, errors='coerce')

        # Perform interpolation on numeric columns
        data[cols_to_convert] = data[cols_to_convert].interpolate(method='linear')

        # Extrapolate when there is only one detection (forward and backward fill)
        data[cols_to_convert] = data[cols_to_convert].fillna(method='ffill').fillna(method='bfill')

        # Add the 'inter_or_extra_polated' column
        data['inter_or_extra_polated'] = np.where(data[cols_to_convert].isnull().any(axis=1), 'yes', 'no')

        # Handle cases where no interpolation is necessary
        data['inter_or_extra_polated'] = np.where(data['frame_number'].isin(full_frame_range['frame_number']) & 
                                                data[cols_to_convert].notnull().all(axis=1), 'no', 'yes')
        
        
        # Export the final DataFrame to CSV
        data.to_csv(output_csv_path, index=False)