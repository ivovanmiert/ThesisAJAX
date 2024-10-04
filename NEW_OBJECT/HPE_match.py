import pandas as pd
import ast 

class HPE_Matcher:
        def __init__(self, hpe_csv_path, hpe_sort):
            """
            Initializes the HPE_Matcher with the path to the HPE CSV file.
            """
            if hpe_sort == 'bottom_up':
                self.hpe_df = pd.read_csv(hpe_csv_path)
                self.hpe_df['Bbox'] = self.hpe_df['Bbox'].apply(ast.literal_eval)
                self.hpe_df[['Keypoint_' + str(i + 1) for i in range(17)]] = self.hpe_df[['Keypoint_' + str(i + 1) for i in range(17)]].applymap(ast.literal_eval)
                self.hpe_df['HPE_ID'] = self.hpe_df.index
                self.hpe_df['Assigned'] = 0
                #print(self.hpe_df)
                self.matched_hpe = {}

            if hpe_sort == 'top_down':
                #print(hpe_csv_path)
                self.hpe_df = pd.read_csv(hpe_csv_path)
                #print('HIER NU EEE:')
                #print(self.hpe_df)
                self.hpe_df['HPE_ID'] = self.hpe_df.index
                self.hpe_df['Assigned'] = 0
                #print(self.hpe_df)
                self.matched_hpe = {}
                 

        def ioui(self, bbox1, bbox2):
            """
            Calculates the Intersection over Union of two bounding boxes.
            """
            x1_min, y1_min, x1_max, y1_max = bbox1
            x2_min, y2_min, x2_max, y2_max = bbox2
            
            xi1 = max(x1_min, x2_min)
            yi1 = max(y1_min, y2_min)
            xi2 = min(x1_max, x2_max)
            yi2 = min(y1_max, y2_max)
            
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            
            bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
            bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = bbox1_area + bbox2_area - inter_area
            return inter_area / union_area if union_area > 0 else 0
        
        def is_hpe_assigned(self, hpe_id):
             """
             Checks if the HPE detection with the given ID has already been assigned. 
             """
             if hpe_id in self.matched_hpe:
                  return True, self.matched_hpe[hpe_id]['iou']
             return False, None 
        
        def assign_hpe(self, hpe_id, player_id, iou):
             """
             Assigns the HPE detection to a player detection
             """
             self.matched_hpe[hpe_id] = {'player_id': player_id, 'iou': iou}
             self.hpe_df.loc[self.hpe_df['HPE_ID'] == hpe_id, 'Assigned'] = 1


        def match_single_detection(self, frame_id, player_bbox):
            """
             Matches a single player detection with the best HPE bounding box, based on IOU, of the specified frame
            """
            hpe_frame = self.hpe_df[self.hpe_df['Frame'] == frame_id + 1].copy() #The +1 since the framenumbers of the pifpaf hpe detections start at frame 1 instead of 0
            hpe_matches = []

            for _, hpe_row in hpe_frame.iterrows():
                bbox_hpe_orig = hpe_row['Bbox']
                bbox_hpe = [
                    bbox_hpe_orig[0],
                    bbox_hpe_orig[1],
                    bbox_hpe_orig[0] + bbox_hpe_orig[2],
                    bbox_hpe_orig[1] + bbox_hpe_orig[3]
                ]
                current_iou = self.ioui(player_bbox, bbox_hpe)
                hpe_coordinates = [hpe_row[f"Keypoint_{i+1}"] for i in range(17)]
                hpe_id = hpe_row['HPE_ID']
                hpe_matches.append((hpe_coordinates, hpe_id, current_iou))

            # Sort the matches by IoU in descending order
            hpe_matches.sort(key=lambda x: x[2], reverse=True)
            return hpe_matches
        
        def match_detection_top_down(self, frame_number, detection_id):
            """
            Returns the HPE coordinates (keypoints) of the row with the corresponding frame_number and detection_id.
            """
            # Filter the dataframe to get the row with the matching frame_number and detection_id
            matched_row = self.hpe_df[(self.hpe_df['frame_number'] == frame_number) & 
                                    (self.hpe_df['detection_id'] == detection_id)]
            
            if matched_row.empty:
                print(f"No match found for Frame Number: {frame_number} and Detection ID: {detection_id}")
                return None

            # Extract the keypoints from the matched row
            keypoints = []
            for i in range(17):
                x_col = f'hpe_keypoint_{i}_x'
                y_col = f'hpe_keypoint_{i}_y'
                
                # Extract the (x, y) coordinates for each keypoint
                x = matched_row.iloc[0][x_col]
                y = matched_row.iloc[0][y_col]
                
                keypoints.append((x, y))

            return keypoints
    
        
        def update_assignment(self, hpe_id, player_id, iou):
             """
             Updates the assignemnt of an HPE detection to a new player detection/
             """
             current_iou = self.matched_hpe[hpe_id]['iou']
             if iou > current_iou:
                self.assign_hpe(hpe_id, player_id, iou)
                return True
             return False
        
        def unassign_hpe(self, hpe_id):
            """
             Unassigns the HPE detection from any player it was previously assigned to. 
            """
            if hpe_id in self.matched_hpe:
                  del self.matched_hpe[hpe_id]
            self.hpe_df.loc[self.hpe_df['HPE_ID'] == hpe_id, 'Assigned'] = 0
        
        def get_hpe_df(self):
            """
            Returns the DataFrame containing HPE data.
            """
            return self.hpe_df

        


