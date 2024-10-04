import math
import numpy as np
import pandas as pd

"""
This file is regarding creating features related to players. 
"""


class PlayerInfo:
    def __init__(self, player_id, players_object, ball_object, field_keypoints):
        """
        Initializes the PlayerInfo object with the given Player ID, Players object, Ball object, and field keypoints.
        """
        self.player_id = player_id
        self.players = players_object
        self.ball = ball_object
        self.field_keypoints = field_keypoints
        self.player = self.players.get_player(self.player_id)
        self.team = self.player.team_classification
        #print('new round:')
        self.location = self.calculate_player_locations()
        self.distances = self.calculate_distances_features()
        self.velocity = self.calculate_velocity_features()
        self.directions = self.calculate_directions_features()
        self.player_ball_features = self.calculate_player_ball_features()
        #self.location_features = self.calculate_location_features()
        self.angle_features = self.calculate_angle_features()
        self.team_features = self.calculate_team_features()
        #self.proximity_features = self.calculate_proximity_features()
        

    def get_pixel_position(self):
        """
        Returns the current pixel coordinates of the player.
        """
        return self.player.pixel_coordinates

    def get_pitch_position(self):
        """
        Returns the current pitch coordinates of the player.
        """
        return self.player.pitch_coordinates

    def get_hpe_position(self):
        """
        Returns the current HPE pixel coordinates of the player.
        """
        return self.player.hpe_pixel_coordinates

    @staticmethod
    def calculate_distance(point1, point2):
        """
        Calculates the Euclidean distance between two points.
        
        :param point1: The first point as a tuple (x, y).
        :param point2: The second point as a tuple (x, y).
        :return: The Euclidean distance between the two points.
        """
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def calculate_velocity(self, distance, frame_diff):
        """
        Calculate the velocity given the distance and frame difference.
        
        :param distance: Distance traveled.
        :param frame_diff: Number of frames over which the distance was traveled.
        :return: Velocity as a tuple (vx, vy).
        """
        #print(distance)
        if frame_diff == 0:
            return (0, 0)
        return (distance / frame_diff, distance / frame_diff)

    def calculate_acceleration(self, frames=[1, 3, 5, 10]):
        """
        Calculate the acceleration of the player over the last given number of frames based on pitch coordinates.

        :param frames: List of frames over which to calculate the acceleration.
        :return: Dictionary with accelerations for each frame count.
        """
        accelerations = {}
        for frame_count in frames:
            if len(self.player.previous_positions) < 2 * frame_count:
                accelerations[frame_count] = (0, 0)  # Not enough data to calculate acceleration
                continue

            velocity_now = self.calculate_velocity([frame_count])[frame_count]
            velocity_before = self.calculate_velocity([2 * frame_count])[2 * frame_count]

            acceleration = ((velocity_now[0] - velocity_before[0]) / frame_count, (velocity_now[1] - velocity_before[1]) / frame_count)
            accelerations[frame_count] = acceleration
        return accelerations

    def calculate_direction(self, frames=[1, 3, 5, 10]):
        """
        Calculate the direction of the player over the last given number of frames based on pitch coordinates.

        :param frames: List of frames over which to calculate the direction.
        :return: Dictionary with directions for each frame count.
        """
        directions = {}
        for frame_count in frames:
            if len(self.player.previous_positions) < frame_count:
                directions[frame_count] = 0  # Not enough data to calculate direction
                continue

            previous_position = self.player.previous_positions[-frame_count]['pitch_coordinates'][0][0]
            current_position = self.player.pitch_coordinates[0][0]

            delta_x = current_position[0] - previous_position[0]
            delta_y = current_position[1] - previous_position[1]

            direction = np.arctan2(delta_y, delta_x) * (180 / np.pi)
            directions[frame_count] = direction
        return directions
    
    def calculate_hpe_keypoint_distances(self, ball_coords_pixel, player):
        """
        Calculates the distance of the ball to every keypoint of the player object.

        :param ball_coords_pixel: The ball's pixel coordinates.
        :param player: The player object.
        :return: A dictionary with distances to player hpe keypoints.
        """
        keypoint_distances = {}
            # Check if player has HPE keypoints and they are not empty
        if not hasattr(player, 'hpe_pixel_coordinates') or not player.hpe_pixel_coordinates:
            return keypoint_distances
        
        for i, keypoint in enumerate(player.hpe_pixel_coordinates):
            dist = self.calculate_distance(ball_coords_pixel, keypoint)
            keypoint_distances[f'distance_ball_to_hpe_keypoint_{i}'] = dist

        return keypoint_distances
    
    def calculate_player_locations(self):
        all_positions = self.player.get_all_positions()
        player_locations = []
        #print('all_positions:')
        #print(all_positions)
        for position in all_positions:
            # print('hierzooo')
            # print(position['pitch_coordinates'])
            # print(position['pixel_coordinates'])
            # print(position['hpe_pixel_coordinates'])
            hpe_positions = position['hpe_pixel_coordinates']
            xmin, ymin, xmax, ymax= position['pixel_coordinates']
            middle_under_pixel = ((xmin + xmax) / 2, ymax)

            # Check if hpe_positions is None and handle it
            if hpe_positions is None:
                # Assign -1 for all hpe keypoint positions
                hpe_positions = [(-1, -1) for _ in range(17)]
            #print('hierzooo nu:')
            #print(f"hpe_positions hierzo: {hpe_positions}")
            #print(f"player id hier: {self.player_id}")
            frame_location = {
                'frame_number': position['frame_number'],
                'x_coordinate_pitch': position['pitch_coordinates'][0,0,0], 
                'y_coordinate_pitch': position['pitch_coordinates'][0,0,1],
                'x_coordinate_pixel': middle_under_pixel[0],
                'y_coordinate_pixel': middle_under_pixel[1], 
                'x_min_coordinate_pixel': xmin,
                'y_min_coordinate_pixel': ymin,
                'x_max_coordinate_pixel': xmax,
                'y_max_coordinate_pixel': ymax,
                'x_coordinate_hpe_keypoint_0': hpe_positions[0][0],
                'y_coordinate_hpe_keypoint_0':hpe_positions[0][1],
                'x_coordinate_hpe_keypoint_1':hpe_positions[1][0],
                'y_coordinate_hpe_keypoint_1':hpe_positions[1][1],
                'x_coordinate_hpe_keypoint_2':hpe_positions[2][0],
                'y_coordinate_hpe_keypoint_2':hpe_positions[2][1],
                'x_coordinate_hpe_keypoint_3':hpe_positions[3][0],
                'y_coordinate_hpe_keypoint_3':hpe_positions[3][1],
                'x_coordinate_hpe_keypoint_4':hpe_positions[4][0],
                'y_coordinate_hpe_keypoint_4':hpe_positions[4][1],
                'x_coordinate_hpe_keypoint_5':hpe_positions[5][0],
                'y_coordinate_hpe_keypoint_5':hpe_positions[5][1],
                'x_coordinate_hpe_keypoint_6':hpe_positions[6][0],
                'y_coordinate_hpe_keypoint_6':hpe_positions[6][1],
                'x_coordinate_hpe_keypoint_7':hpe_positions[7][0],
                'y_coordinate_hpe_keypoint_7':hpe_positions[7][1],
                'x_coordinate_hpe_keypoint_8':hpe_positions[8][0],
                'y_coordinate_hpe_keypoint_8':hpe_positions[8][1],
                'x_coordinate_hpe_keypoint_9':hpe_positions[9][0],
                'y_coordinate_hpe_keypoint_9':hpe_positions[9][1],
                'x_coordinate_hpe_keypoint_10':hpe_positions[10][0],
                'y_coordinate_hpe_keypoint_10':hpe_positions[10][1],
                'x_coordinate_hpe_keypoint_11':hpe_positions[11][0],
                'y_coordinate_hpe_keypoint_11':hpe_positions[11][1],
                'x_coordinate_hpe_keypoint_12':hpe_positions[12][0],
                'y_coordinate_hpe_keypoint_12':hpe_positions[12][1],
                'x_coordinate_hpe_keypoint_13':hpe_positions[13][0],
                'y_coordinate_hpe_keypoint_13':hpe_positions[13][1],
                'x_coordinate_hpe_keypoint_14':hpe_positions[14][0],
                'y_coordinate_hpe_keypoint_14':hpe_positions[14][1],
                'x_coordinate_hpe_keypoint_15':hpe_positions[15][0],
                'y_coordinate_hpe_keypoint_15':hpe_positions[15][1],
                'x_coordinate_hpe_keypoint_16':hpe_positions[16][0],
                'y_coordinate_hpe_keypoint_16':hpe_positions[16][1],
                
            }
            player_locations.append(frame_location)
        return player_locations
    
    def get_player_hpe_keypoints_locations(self):
        all_positions = self.player.get_all_positions()



    def calculate_distances_features(self):
        """
        Calculates the distance from the player to each keypoint for every frame.
        
        :return: A list of dictionaries, where each dictionary contains distances to keypoints for a single frame.
        """
        all_positions = self.player.get_all_positions()
        #print(all_positions)
        distances = []

        # List of keypoint indices we are interested in
        keypoints_of_interest = [2, 3, 26, 27, 8, 9, 16, 17, 42, 12, 13, 28, 29]
        #print(all_positions)
        for position in all_positions:
            frame_distances = {'frame_number': position['frame_number']}
            player_coords = position['pitch_coordinates'][0][0]
            for name, coords in self.field_keypoints.get_keypoints().items():
                if name in keypoints_of_interest:
                    distance = self.calculate_distance(player_coords, coords)
                    frame_distances[f'distance_to_field_keypoint_{name}'] = distance
            distances.append(frame_distances)

        return distances

    def calculate_velocity_features(self):
        """
        Calculates the features: distance traveled, velocity, and acceleration over different frame intervals.
        Intervals chosen are 1 frame, 3 frames, 5 frames, 10 frames and 20 frames.
        :return: A dictionary containing the features for each frame.
        """
        all_positions = self.player.get_all_positions()
        features = []
        intervals = [1, 3, 5, 10, 20]

        for i, current_position in enumerate(all_positions):
            frame_features = {'frame_number': current_position['frame_number']}
            current_coords = current_position['pitch_coordinates'][0][0]

            for interval in intervals:
                if i >= interval:
                    previous_position = all_positions[i - interval]
                    previous_coords = previous_position['pitch_coordinates'][0][0]
                    distance = self.calculate_distance(current_coords, previous_coords)
                    frame_diff = current_position['frame_number'] - previous_position['frame_number']
                    
                    velocity = self.calculate_velocity(distance, frame_diff)
                    
                    if i >= 2 * interval:
                        earlier_position = all_positions[i - 2 * interval]
                        earlier_coords = earlier_position['pitch_coordinates'][0][0]
                        earlier_distance = self.calculate_distance(previous_coords, earlier_coords)
                        earlier_frame_diff = previous_position['frame_number'] - earlier_position['frame_number']
                        
                        earlier_velocity = self.calculate_velocity(earlier_distance, earlier_frame_diff)
                        acceleration = ((velocity[0] - earlier_velocity[0]) / interval, (velocity[1] - earlier_velocity[1]) / interval)
                    else:
                        acceleration = None

                    frame_features[f'distance_traveled_{interval}_frames'] = distance
                    frame_features[f'velocity_{interval}_frames'] = velocity
                    frame_features[f'acceleration_{interval}_frames'] = acceleration
                else:
                    frame_features[f'distance_traveled_{interval}_frames'] = None
                    frame_features[f'velocity_{interval}_frames'] = None
                    frame_features[f'acceleration_{interval}_frames'] = None

            features.append(frame_features)

        return features

    def calculate_directions_features(self):
        """
        Calculates the player direction over different frame intervals.

        :return: A dictionary containing the directions for each frame.
        """
        all_positions = self.player.get_all_positions()
        directions = []
        intervals = [1, 3, 5, 10]

        for i, current_position in enumerate(all_positions):
            frame_directions = {'frame_number': current_position['frame_number']}
            current_coords = current_position['pitch_coordinates'][0][0]

            for interval in intervals:
                if i >= interval:
                    previous_position = all_positions[i - interval]
                    previous_coords = previous_position['pitch_coordinates'][0][0]
                    delta_x = current_coords[0] - previous_coords[0]
                    delta_y = current_coords[1] - previous_coords[1]
                    direction = np.arctan2(delta_y, delta_x) * (180 / np.pi)
                    frame_directions[f'direction_{interval}_frames'] = direction
                else:
                    frame_directions[f'direction_{interval}_frames'] = None

            directions.append(frame_directions)

        return directions

    def calculate_player_ball_features(self):
        """
        Calculates the features related to the player's interaction with the ball.
        
        :return: A dictionary containing player-ball related features, namely the distance to the ball in pitch coordinates, and the distance to the ball in pixel coordinates.
        """
        ball_positions = self.ball.get_all_positions()
        player_ball_features = []

        for position in ball_positions:
            frame_number = position['frame_number']
            ball_pitch_position = position['pitch_coordinates'][0][0]
            ball_pixel_position = position['pixel_coordinates']
            # print('hierzo:')
            player_position_info = self.player.get_position_at_frame(frame_number)
            if player_position_info is not None:
                # print('hello')
                pitch_coordinates = player_position_info['pitch_coordinates'][0][0]
                pixel_coordinates = player_position_info['pixel_coordinates']
                xmin, ymin, xmax, ymax = pixel_coordinates
                middle_upper_pixel = ((xmin + xmax) / 2, ymin)
                player_ball_distance_pitch = self.calculate_distance(pitch_coordinates, ball_pitch_position)
                player_ball_distance_pixel = self.calculate_distance(middle_upper_pixel, ball_pixel_position)
                # Calculate distances from HPE keypoints to the ball
                hpe_keypoint_distances = self.calculate_hpe_keypoint_distances(ball_pixel_position, self.player)

                frame_features = {
                    'frame_number': frame_number,
                    'distance_to_ball_pitch': player_ball_distance_pitch,
                    'distance_to_ball_pixel': player_ball_distance_pixel,
                    **hpe_keypoint_distances #'distance_to_ball_hpe': hpe_keypoint_distances

                }

                player_ball_features.append(frame_features)

        return player_ball_features
    
    def calculate_team_features(self):
        """
        Converts the team classification into one-hot encoding for each frame.
        
        :return: A list of dictionaries where each dictionary contains one-hot encoded team features for a single frame.
        """
        # Define the possible teams for one-hot encoding
        team_classes = ["Team_1", "Team_2", "Goalkeeper_Team_1", "Goalkeeper_Team_2", "Referee"]
        
        # Initialize a list to store team features for all frames
        team_features = []
        
        # Retrieve all player positions for consistency with other features
        all_positions = self.player.get_all_positions()
        
        # Get the team classification of the player
        player_team = self.player.team_classification
        
        # Iterate over all player positions to create the team features
        for position in all_positions:
            frame_number = position['frame_number']
            
            # Initialize the one-hot encoded feature dictionary for the current frame
            frame_features = {'frame_number': frame_number}
            
            # Set the one-hot encoding for each team
            for team in team_classes:
                frame_features[f'team_{team}'] = 1 if player_team == team else 0
            
            # Append the frame-specific team features to the list
            team_features.append(frame_features)
        
        return team_features


    def calculate_proximity_features(self):
        """
        Calculates the proximity features to all other players.
        
        :return: A dictionary containing proximity features to other players.
        """
        proximity_features = []

        for position in self.player.get_all_positions():
            frame_number = position['frame_number']
            player_pitch_position = position['pitch_coordinates'][0][0]

            frame_features = {'frame_number': frame_number}
            for other_player in self.players.get_all_players():
                if other_player.player_id == self.player_id:
                    continue
                
                other_player_position = other_player.get_position_at_frame(frame_number)
                if other_player_position is None:
                    continue
                #print('other_player_position')
                #print(other_player_position)
                distance_to_other_player = self.calculate_distance(player_pitch_position, other_player_position['pitch_coordinates'][0][0])
                frame_features[f'distance_to_player_{other_player.player_id}'] = distance_to_other_player

            proximity_features.append(frame_features)

        return proximity_features

    def calculate_angle_features(self):
        """
        Calculate angular features of the player with respect to the ball and other players.
        
        :return: A list of dictionaries, where each dictionary contains angular features for a single frame.
        """
        all_positions = self.player.get_all_positions()
        ball_positions = self.ball.get_all_positions()
        angle_features = []

        for player_position in all_positions:
            frame_features = {'frame_number': player_position['frame_number']}
            player_coords = player_position['pitch_coordinates'][0][0]
            player_coords_pixel = player_position['pixel_coordinates']
            #print(f"player coords pixel: {player_coords_pixel}")
            # Find the corresponding ball position for the current frame
            ball_position = next((bp for bp in ball_positions if bp['frame_number'] == player_position['frame_number']), None)
            if ball_position:
                #print(f"ball position: {ball_position}")
                ball_coords = ball_position['pitch_coordinates'][0][0]
                #print(f"ball coords: {ball_coords}")
                # Angle between player and ball
                delta_x = ball_coords[0] - player_coords[0]
                delta_y = ball_coords[1] - player_coords[1]
                angle_to_ball = np.arctan2(delta_y, delta_x) * (180 / np.pi)
                frame_features['angle_to_ball'] = angle_to_ball

                ball_coords_pixel = ball_position['pixel_coordinates']
                #print(f"ball coords pixel: {ball_coords_pixel}")
                # Calculate the center x and maximum y pixel for the ball
                player_center_x_pixel = (player_coords_pixel[0] + player_coords_pixel[2]) / 2
                player_max_y_pixel = player_coords_pixel[3]
                #print(f"deze 2: {player_center_x_pixel, player_max_y_pixel}")
                # Angle between player and ball (pixel coordinates)
                delta_x_pixel = ball_coords_pixel[0] - player_center_x_pixel
                delta_y_pixel = ball_coords_pixel[1] - player_max_y_pixel
                angle_to_ball_pixel = np.arctan2(delta_y_pixel, delta_x_pixel) * (180 / np.pi)
                frame_features['angle_to_ball_pixels'] = angle_to_ball_pixel          
            else:
                frame_features['angle_to_ball'] = None
                frame_features['angle_to_ball_pixels'] = None
            other_players = [p for p in self.players.get_all_players() if p.player_id != self.player_id]
            angles_to_others = []

            for other_player in other_players:
                other_positions = other_player.get_all_positions()
                other_position = next((op for op in other_positions if op['frame_number'] == player_position['frame_number']), None)
                if other_position:
                    other_coords = other_position['pitch_coordinates'][0][0]
                    delta_x = other_coords[0] - player_coords[0]
                    delta_y = other_coords[1] - player_coords[1]
                    angle_to_other = np.arctan2(delta_y, delta_x) * (180 / np.pi)
                    angles_to_others.append(angle_to_other)

            if angles_to_others:
                min_angle = min(angles_to_others)
                max_angle = max(angles_to_others)
                avg_angle = sum(angles_to_others) / len(angles_to_others)
            else:
                min_angle = None
                max_angle = None
                avg_angle = None

            frame_features['min_angle_to_other_players'] = min_angle
            frame_features['max_angle_to_other_players'] = max_angle
            frame_features['avg_angle_to_other_players'] = avg_angle

            # Adding direction features
            directions = self.calculate_direction()
            for frame_count, direction in directions.items():
                frame_features[f'direction_{frame_count}_frames'] = direction

            angle_features.append(frame_features)

        return angle_features

    def calculate_location_features(self):
        """
        Calculates the location features (pixel and pitch coordinates) for the player.
        
        :return: A list of dictionaries, where each dictionary contains location features for a single frame.
        """
        all_positions = self.player.get_all_positions()
        location_features = []

        for position in all_positions:
            frame_features = {
                'frame_number': position['frame_number'],
                'pixel_coordinates': position['pixel_coordinates'],
                'pitch_coordinates': position['pitch_coordinates'][0][0]
            }
            #print(f"frame_features: met pixel location: {frame_features}")
            location_features.append(frame_features)

        return location_features
    
    def get_all_features(self):
        all_features = {}

        # List of all feature lists
        feature_lists = [self.location, self.distances, self.velocity, self.directions, self.player_ball_features, self.angle_features, self.team_features] #self.proximity_features ff weggelaten

        # Populate the all_features dictionary with frame numbers
        for feature_list in feature_lists:
            for feature in feature_list:
                frame_number = feature['frame_number']
                if frame_number not in all_features:
                    all_features[frame_number] = {'frame_number': frame_number, 'ID': self.player_id}

        # Merge all feature lists into the all_features dictionary
        for feature_list in feature_lists:
            for feature in feature_list:
                frame_number = feature['frame_number']
                all_features[frame_number].update(feature)

        # Convert the dictionary to a list sorted by frame_number
        combined_features = [all_features[frame_number] for frame_number in sorted(all_features)]

        # Convert the combined features to a DataFrame
        df = pd.DataFrame(combined_features)
        #df.to_csv('example_df', index=False)
        return df