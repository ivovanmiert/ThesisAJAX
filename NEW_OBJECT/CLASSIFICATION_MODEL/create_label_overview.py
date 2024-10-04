import os
import pandas as pd
import numpy as np


"""
This file creates the labels of the events. Primary label and deeper detailed labels. 
"""

def create_label_and_length_dataframe(event_folder_path, df, output_path):
    df = pd.read_csv(df)
    print(df)
    df['event_id'] = df['event_id'].astype(int)
    event_ids = [filename.replace('.csv', '') for filename in os.listdir(event_folder_path) if filename.endswith('.csv')]


    data = []


    for event_id in event_ids:
        event_id_int = int(event_id)

        #PRIMARY TYPE:
        # Get the primary_type for the current event_id
        primary_type = df[df['event_id'] == event_id_int]['primary_type'].values[0]

        # Determine the label based on the primary_type
        label_dict = {
            'shot': 0,
            'pass': 1,
            'duel': 2,
            'interception': 3,
            'touch': 4
        }
        label = label_dict.get(primary_type, -1)  # Default to -1 if primary_type is not found
        
        #DUEL TYPE:
        # Initialize the duel label
        duel_label = -1  # Default to 0 (not a duel)

        # Check the columns for duel types
        aerial_duel = df[df['event_id'] == event_id_int]['aerial_duel'].values[0]
        loose_ball_duel = df[df['event_id'] == event_id_int]['loose_ball_duel'].values[0]
        offensive_duel = df[df['event_id'] == event_id_int]['offensive_duel'].values[0]
        defensive_duel = df[df['event_id'] == event_id_int]['defensive_duel'].values[0]

        # Assign duel_label based on the criteria
        if aerial_duel:
            duel_label = 0
        elif loose_ball_duel:
            duel_label = 1
        elif offensive_duel or defensive_duel:
            duel_label = 2
        
        #PASS:
        #KIND OF PASS (CROSS/NORMAL PASS)
        pass_label = -1
        cross_accurate = -1
        cross_direction = -1
        cross_flank = -1
        
        pass_accurate = -1
        pass_direction = -1
        pass_distance = -1
        pass_progressive = -1
        pass_through = -1

        normal_pass = df[df['event_id'] == event_id_int]['normal_pass'].values[0] if pd.notna(df[df['event_id'] == event_id_int]['normal_pass'].values[0]) else None
        cross = df[df['event_id'] == event_id_int]['cross'].values[0] if pd.notna(df[df['event_id'] == event_id_int]['cross'].values[0]) else None
        if normal_pass:
            pass_label = 0
            accuracy_pass = df[df['event_id'] == event_id_int]['pass_accurate'].values[0]
            if accuracy_pass is True:
                pass_accurate = 0
            if accuracy_pass is False:
                pass_accurate = 1
            
            forward = df[df['event_id'] == event_id_int]['forward_pass'].values[0]
            backward = df[df['event_id'] == event_id_int]['back_pass'].values[0]
            lateral = df[df['event_id'] == event_id_int]['lateral_pass'].values[0]
            if forward:
                pass_direction = 0
            elif backward:
                pass_direction = 1
            elif lateral:
                pass_direction = 2

            long = df[df['event_id'] == event_id_int]['long_pass'].values[0]
            short_or_medium = df[df['event_id'] == event_id_int]['short_or_medium_pass'].values[0]
            if long:
                pass_distance = 0
            if short_or_medium:
                pass_distance = 1
            
            progressive = df[df['event_id'] == event_id_int]['progressive_pass'].values[0]
            if progressive:
                pass_progressive = 0
            else:
                pass_progressive = 1
            
            through = df[df['event_id'] == event_id_int]['forward_pass'].values[0]
            if through:
                pass_through = 0
            else: 
                pass_through = 1

        if cross:
            pass_label = 1
            accuracy_cross = df[df['event_id'] == event_id_int]['accurate'].values[0]
            if accuracy_cross:
                cross_accurate = 0
            else:
                cross_accurate = 1

            forward = df[df['event_id'] == event_id_int]['forward_pass'].values[0]
            backward = df[df['event_id'] == event_id_int]['back_pass'].values[0]
            lateral = df[df['event_id'] == event_id_int]['lateral_pass'].values[0]
            if forward:
                cross_direction = 0
            elif backward:
                cross_direction = 1
            elif lateral:
                cross_direction = 2
            
            right = df[df['event_id'] == event_id_int]['flank_right'].values[0]
            left = df[df['event_id'] == event_id_int]['flank_left'].values[0]
            center = df[df['event_id'] == event_id_int]['flank_center'].values[0]
            if right:
                cross_flank = 0
            elif left:
                cross_flank = 1
            elif center:
                cross_flank = 2
        
        #SHOT 
        on_target = -1
        goal = -1
        shot_body_part = -1
        if label == 0:
            on_target_value = df[df['event_id'] == event_id_int]['on_target'].values[0]
            if on_target_value:
                on_target = 0
                goal_value = df[df['event_id'] == event_id_int]['is_goal'].values[0]
                if goal_value:
                    goal = 0
                else:
                    goal = 1
            else:
                on_target = 1
            right_foot = df[df['event_id'] == event_id_int]['right_foot_true'].values[0]
            left_foot = df[df['event_id'] == event_id_int]['left_foot_true'].values[0]
            head_or_other = df[df['event_id'] == event_id_int]['head_or_other_true'].values[0]
            if right_foot:
                shot_body_part = 0
            elif left_foot:
                shot_body_part = 1
            elif head_or_other:
                shot_body_part = 2

        # Append the data to the list
        data.append({'clip_id': event_id, 'label': label, 'body_part_label': shot_body_part, 'duel_label': duel_label, 'pass_label': pass_label, 'cross_accurate': cross_accurate, 'cross_direction': cross_direction, 'cross_flank': cross_flank, 'pass_accurate': pass_accurate, 'pass_direction': pass_direction, 'pass_distance': pass_distance, 'pass_progressive': pass_progressive, 'pass_through': pass_through, 'shot_on_target': on_target, 'shot_goal': goal})

    # Step 4: Create the new dataframe
    df_label_and_length = pd.DataFrame(data)

    # Step 5: Save the new dataframe to a CSV file
    output_file = os.path.join(output_path, 'df_label_full.csv')
    df_label_and_length.to_csv(output_file, index=False)
    print(df_label_and_length)
    print(f"New dataframe created and saved as {output_file}")