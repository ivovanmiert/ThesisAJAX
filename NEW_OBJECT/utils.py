import pandas as pd



def load_detection_df(file_path):
    df = pd.read_csv(file_path, header=None, names=['frame_id', 'track_id', 'x_min', 'y_min', 'x_max', 'y_max', '-1', '-2', '-3', '-4'])
    grouped = df.groupby('frame_id')
    
    def transform_row(group):
        return [
            [row['track_id'], [row['x_min'], row['y_min'], row['x_max'], row['y_max']]]
            for idx, row in group.iterrows()
        ]
    
    transformed_data = grouped.apply(transform_row)
    processed_df = pd.DataFrame({
        'frame_id': transformed_data.index,
        'player_detections': transformed_data.values
    }).reset_index(drop=True)
    
    return processed_df

def old_load_ball_df(file_path):
    df = pd.read_csv(file_path, header=None, names=['Image Path', 'X coordinate', 'Y coordinate', 'Visibility', 'Score'])
    # Extract frame_id from the image path
    df['frame_id'] = df['Image Path'].apply(lambda x: int(x.split('/')[-1].split('_')[1].split('.')[0]))
    # Round the coordinates to two decimal places
    df['X coordinate'] = df['X coordinate'].round(2)
    df['Y coordinate'] = df['Y coordinate'].round(2)
    return df

def load_ball_df(file_path):
    # Load the new CSV file
    print(f"filepath: {file_path}")
    df = pd.read_csv(file_path, header=0) #names=['frame_number', 'class', 'x_center', 'y_center', 'width', 'height', 'confidence'])
    print(df)
    
    # Ensure 'frame_number' is an integer
    df['frame_number'] = df['frame_number'].astype(int)

    # Round the coordinates and dimensions to two decimal places
    df['x_center'] = ((df['x_min'] + df['x_max'])/2).round(2)
    df['y_center'] = ((df['y_min'] + df['y_max'])/2).round(2)
    df['width'] = (df['x_max'] - df['x_min']).round(2)
    df['height'] = (df['y_max'] - df['y_min']).round(2)
    df['confidence'] = df['confidence'].fillna(0).round(2)
    
    return df

def rescale_bbox(bbox, original_scale, target_scale):
    """
    Rescale bounding box coordinates from one scale to another.
    """
    x1, y1, x2, y2 = bbox
    orig_w, orig_h = original_scale
    target_w, target_h = target_scale
    
    x1_rescaled = x1 * target_w / orig_w
    y1_rescaled = y1 * target_h / orig_h
    x2_rescaled = x2 * target_w / orig_w
    y2_rescaled = y2 * target_h / orig_h
    
    return (x1_rescaled, y1_rescaled, x2_rescaled, y2_rescaled)

def calculate_planar_coordinates(player_detection_bbox, frame_id, homography_estimator, player_id):
    original_scale = (1280, 720)
    target_scale = (1280, 720)
    
   
    rescaled_bbox = rescale_bbox(player_detection_bbox, original_scale, target_scale)
    center_bottom_rescaled = ((rescaled_bbox[0] + rescaled_bbox[2]) / 2, rescaled_bbox[3]) #It takes y_max but since the pixels axis starts in the top-left corner with y = 0, for the center of the downside of the bounding box, the max y value should be taken. 
    planar_point = homography_estimator.warp_points(frame_id, center_bottom_rescaled, player_id)
    return planar_point

def rescale_point(point, original_scale, target_scale):
    """
    Rescale point coordinates from one scale to another.
    """
    x, y = point
    orig_w, orig_h = original_scale
    target_w, target_h = target_scale
    
    x_rescaled = x * target_w / orig_w
    y_rescaled = y * target_h / orig_h
    
    return (x_rescaled, y_rescaled)

def calculate_planar_coordinates_ball(ball_detection, frame_id, homography_estimator):
    original_scale = (1280, 720)
    target_scale = (960, 540)
    
    rescaled_point = rescale_point(ball_detection, original_scale, target_scale)
    print(f"frame_id: {frame_id}")
    planar_point = homography_estimator.warp_points(frame_id, rescaled_point, player_id=None)
    return planar_point
