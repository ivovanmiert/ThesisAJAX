import math
import matplotlib.pyplot as plt
import pandas as pd

"""
This class is about creating the features related to the ball. It uses the ball object created, together with the player objects stored in Players. 

"""


class BallInfo:
    def __init__(self, ball, field_keypoints, players):
        """
        Initializes the BallInfo object with the given ball and field keypoints.
        """
        self.ball = ball
        self.field_keypoints = field_keypoints #This field_keypoints must already be the right field keypoint object (so it has to be inserted from the very beginning)
        self.players = players
        self.distances = self.calculate_distances_features()
        self.velocity = self.calculate_velocity_features()
        self.directions = self.calculate_directions_features()
        self.area_checks = self.calculate_area_checks()
        self.trajectory_analysis = self.analyze_trajectory()
        self.ball_locations = self.calculate_ball_locations()
        self.ball_player_features = self.calculate_ball_player_features()

    @staticmethod
    def calculate_distance(point1, point2):
        """
        Calculates the Euclidean distance between two points.
        """
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_velocity(self, distance, frame_diff):
        """
        Calculate the velocity given a distance and frame difference.
        """
        return distance / frame_diff
    
    def calculate_acceleration(self, velocity1, velocity2, frame_diff):
        """
        Calculate the acceleration given two velocities and frame difference.
        """
        return (velocity2 - velocity1) / frame_diff 
    
    def calculate_direction(self, pos1, pos2):
        """
        Calculate the direction from pos1 to pos2 in degrees.
        """
        delta_x = pos2[0] - pos1[0]
        delta_y = pos2[1] - pos1[1]
        angle = math.degrees(math.atan2(delta_y, delta_x))
        return angle if angle >= 0 else 360 + angle
    
    def calculate_ball_locations(self):
        all_positions = self.ball.get_all_positions()
        ball_locations = []
        for position in all_positions:
            frame_location = {
                'frame_number': position['frame_number'],
                'x_coordinate_pitch': position['pitch_coordinates'][0][0][0],
                'y_coordinate_pitch': position['pitch_coordinates'][0][0][1],
                'x_coordinate_pixel': position['pixel_coordinates'][0],
                'y_coordinate_pixel': position['pixel_coordinates'][1]
            }
            ball_locations.append(frame_location)

        return ball_locations
    
    def count_players_within_distances(self, ball_coords):
        """
        Counts the number of players within specified distances.
        """
        
        distances = [0.2, 0.5, 1, 3, 5]
        counts = {f'players_within_{d}m': 0 for d in distances}
        counts.update({f'players_teamA_within_{d}m': 0 for d in distances})
        counts.update({f'players_teamB_within_{d}m': 0 for d in distances})

        for player in self.players.players_dict.values():
            player_field_coords = player.pitch_coordinates
            player_field_coords = (player_field_coords[0][0][0], player_field_coords[0][0][1])
            dist = self.calculate_distance(ball_coords, player_field_coords)

            for d in distances:
                if dist <= d:
                    counts[f'players_within_{d}m'] += 1
                    if player.team_classification == 'Team_1':
                        counts[f'players_teamA_within_{d}m'] += 1
                    elif player.team_classification == 'Team_2':
                        counts[f'players_teamB_within_{d}m'] += 1
        return counts
    
    def calculate_hpe_keypoint_distances(self, ball_coords_pixel, player):
        """
        Calculates the distance of the ball to every keypoint of the closest player object if within 2 meters.
        """
        keypoint_distances = {}
        for i, keypoint in enumerate(player.hpe_pixel_coordinates):
            dist = self.calculate_distance(ball_coords_pixel, keypoint)
            keypoint_distances[f'distance_to_keypoint_{i}'] = dist

        return keypoint_distances
    

    
    def calculate_distances_features(self):
        """
        Calculates the distance from the ball to specified keypoints for every frame.
        """
        all_positions = self.ball.get_all_positions()
        distances = []

        keypoints_of_interest = [2, 3, 26, 27, 8, 9, 16, 17, 42, 12, 13, 28, 29]

        for position in all_positions:
            frame_distances = {'frame_number': position['frame_number']}
            ball_coords = position['pitch_coordinates'][0][0]

            for name, coords in self.field_keypoints.get_keypoints().items():
                if name in keypoints_of_interest:
                    distance = self.calculate_distance(ball_coords, coords)
                    frame_distances[f'distance_to_field_keypoint_{name}'] = distance

            distances.append(frame_distances)

        return distances
    
    def calculate_velocity_features(self):
        """
        Calculates the features: distance traveled, velocity, and acceleration over different frame intervals.
        Intervals chosen are 1 frame, 3 frames, 5 frames, 10 frames and 20 frames.
        """
        all_positions = self.ball.get_all_positions()
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
                        acceleration = self.calculate_acceleration(earlier_velocity, velocity, interval)
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
        Calculates the ball direction over different frame intervals.
        """
        all_positions = self.ball.get_all_positions()
        directions = []
        intervals = [1, 3, 5, 10]

        for i, current_position in enumerate(all_positions):
            frame_directions = {'frame_number': current_position['frame_number']}
            current_coords = current_position['pitch_coordinates'][0][0]

            for interval in intervals:
                if i >= interval:
                    previous_position = all_positions[i - interval]
                    previous_coords = previous_position['pitch_coordinates'][0][0]
                    direction = self.calculate_direction(previous_coords, current_coords)
                    frame_directions[f'direction_{interval}_frames'] = direction
                else:
                    frame_directions[f'direction_{interval}_frames'] = None

            directions.append(frame_directions)

        return directions
    

    def calculate_area_checks(self):
        """
        Checks in which area the ball is according to its pitch_coordinates
        """

        all_positions = self.ball.get_all_positions()
        area_checks = []

        for position in all_positions:
            frame_checks = {'frame_number': position['frame_number']}
            ball_coords = position['pitch_coordinates'][0][0]
            
            frame_checks['in_middle_third'] = self.field_keypoints.is_in_middle_third(ball_coords)
            frame_checks['in_left_penalty_area'] = self.field_keypoints.is_in_left_penalty_area(ball_coords)
            frame_checks['in_right_penalty_area'] = self.field_keypoints.is_in_right_penalty_area(ball_coords)
            frame_checks['in_left_deep_completion_area'] = self.field_keypoints.is_in_left_deep_completion_area(ball_coords)
            frame_checks['in_right_deep_completion_area'] = self.field_keypoints.is_in_right_deep_completion_area(ball_coords)

            area_checks.append(frame_checks)

        return area_checks
    
    def analyze_trajectory(self):
        all_positions = self.ball.get_all_positions()
        trajectory_analysis = []
        intervals = [1, 3, 5, 10]

        for i, position in enumerate(all_positions):
            frame_trajectory = {'frame_number': position['frame_number']}
            ball_coords = position['pitch_coordinates'][0][0]
            
            for interval in intervals:
                if i >= interval:
                    previous_position = all_positions[i - interval]
                    previous_coords = previous_position['pitch_coordinates'][0][0]
                    direction = self.calculate_direction(previous_coords, ball_coords)
                    frame_trajectory[f'trajectory_direction_{interval}_frames'] = direction
                    
                    consistency = self.check_trajectory_consistency(i, interval, all_positions)
                    frame_trajectory[f'consistency_{interval}_frames'] = consistency
                    
                    if consistency == 0:
                        frame_trajectory[f'consistency_{interval}_frames_straight'] = 1
                        frame_trajectory[f'consistency_{interval}_frames_semistraight'] = 0
                        frame_trajectory[f'consistency_{interval}_frames_notstraight'] = 0
                    elif consistency == 1:
                        frame_trajectory[f'consistency_{interval}_frames_straight'] = 0
                        frame_trajectory[f'consistency_{interval}_frames_semistraight'] = 1
                        frame_trajectory[f'consistency_{interval}_frames_notstraight'] = 0
                    elif consistency == 2:
                        frame_trajectory[f'consistency_{interval}_frames_straight'] = 0
                        frame_trajectory[f'consistency_{interval}_frames_semistraight'] = 0
                        frame_trajectory[f'consistency_{interval}_frames_notstraight'] = 1
                    else:
                        frame_trajectory[f'consistency_{interval}_frames_straight'] = None
                        frame_trajectory[f'consistency_{interval}_frames_semistraight'] = None
                        frame_trajectory[f'consistency_{interval}_frames_notstraight'] = None
                else:
                    frame_trajectory[f'trajectory_direction_{interval}_frames'] = None
                    frame_trajectory[f'consistency_{interval}_frames'] = None
                    frame_trajectory[f'consistency_{interval}_frames_straight'] = None
                    frame_trajectory[f'consistency_{interval}_frames_semistraight'] = None
                    frame_trajectory[f'consistency_{interval}_frames_notstraight'] = None

            trajectory_analysis.append(frame_trajectory)

        return trajectory_analysis

    def check_trajectory_consistency(self, current_index, interval, all_positions):
        current_position = all_positions[current_index]
        current_coords = current_position['pitch_coordinates'][0][0]
        previous_position = all_positions[current_index - interval]
        previous_coords = previous_position['pitch_coordinates'][0][0]
        direction1 = self.calculate_direction(previous_coords, current_coords)

        if current_index >= 2 * interval:
            earlier_position = all_positions[current_index - 2 * interval]
            earlier_coords = earlier_position['pitch_coordinates'][0][0]
            direction2 = self.calculate_direction(earlier_coords, previous_coords)
            direction_change = abs(direction1 - direction2)
            
            if direction_change < 5:
                return 0
            elif direction_change < 15:
                return 1
            else:
                return 2
        else:
            return None
        
    def visualize_trajectory(self):
        all_positions = self.ball.get_all_positions()
        x_coords = [pos['pitch_coordinates'][0][0][0] for pos in all_positions]
        y_coords = [pos['pitch_coordinates'][0][0][1] for pos in all_positions]
        
        fig, ax = plt.subplots()
        ax.plot(x_coords, y_coords, marker='o')

        for i, pos in enumerate(all_positions):
            consistency_labels = [f'{interval}_frames: {self.trajectory_analysis[i][f"consistency_{interval}_frames"]}' for interval in [1, 3, 5, 10]]
            label = f'Frame {pos["frame_number"]}\n' + '\n'.join(consistency_labels)
            ax.annotate(label, (x_coords[i], y_coords[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

        ax.set_xlabel('X Coordinates')
        ax.set_ylabel('Y Coordinates')
        ax.set_title('Ball Trajectory with Consistency Analysis')
        plt.show()


    def calculate_ball_player_features(self):
        """
        Calculates features based on the ball-player connections.
        """
        all_positions = self.ball.get_all_positions()
        ball_player_features = []

        for position in all_positions:
            frame_features = {'frame_number': position['frame_number']}
            ball_coords_pixel = position['pixel_coordinates']
            ball_coords_field = position['pitch_coordinates'][0][0]

            #Get players distances in both pixel and field coordinates
            player_distances = []
            for player in self.players.players_dict.values():
                player_pixel_coords = player.pixel_coordinates
                player_field_coords = player.pitch_coordinates

                dist_pixel = self.calculate_distance(ball_coords_pixel, player_pixel_coords)
                player_field_coords = (player_field_coords[0][0][0], player_field_coords[0][0][1])
                dist_field = self.calculate_distance(ball_coords_field, player_field_coords)
                player_distances.append((dist_pixel, dist_field, player.team_classification, player.player_id))

            #Sort players by distance in pixel coordinates
            player_distances.sort(key=lambda x: x[0])
            closest_pixel_distances = player_distances[:5]

            for i, (dist_pixel, dist_field, team, id) in enumerate(closest_pixel_distances):
                frame_features[f'closest_pixel_distance_{i+1}'] = dist_pixel
                frame_features[f'closest_pixel_distance_field_{i+1}'] = dist_field
                frame_features[f'closest_pixel_distance_team_{i+1}'] = team
                frame_features[f'closest_pixel_distance_id_{i+1}'] = id

            # Sort players by distance in field coordinates
            player_distances.sort(key=lambda x: x[1])
            closest_field_distances = player_distances[:5]

            for i, (dist_pixel, dist_field, team, id) in enumerate(closest_field_distances):
                frame_features[f'closest_field_distance_{i+1}'] = dist_field #dit veranderd, eerst stond hier dist_pixel
                frame_features[f'closest_field_distance_pixel_{i+1}'] = dist_pixel
                frame_features[f'closest_field_distance_team_{i+1}'] = team
                frame_features[f'closest_pixel_distance_id_{i+1}'] = id

        
            # Count players within specific distances
            frame_features.update(self.count_players_within_distances(ball_coords_field))

            # Distance to keypoints of the closest player within 2 meters
            closest_player_within_2m = [p for p in player_distances if p[1] <= 2]
            #print('closest_player_within_2m:')
            #print(closest_player_within_2m)
            if closest_player_within_2m:
                closest_player = self.players.get_player(closest_player_within_2m[0][3])
                #print('closest_player:')
                #print(closest_player)
                for team in ['Team_1', 'Team_2', 'Goalkeeper_Team_1', 'Goalkeeper_Team_2', 'Referee']:
                    if closest_player.team_classification == team:
                        keypoint_distances = self.calculate_hpe_keypoint_distances(ball_coords_pixel, closest_player)
                        frame_features.update(keypoint_distances)

            ball_player_features.append(frame_features)

        return ball_player_features
    
    def get_features_for_lstm(self):
        return {
            'distances': self.distances,
            'velocity': self.velocity,
            'directions': self.directions,
            'area_checks': self.area_checks,
            'trajectory_analysis': self.trajectory_analysis,
            'ball_locations': self.ball_locations,
            'ball_player_features': self.ball_player_features
        }
    
    def get_features_for_specific_frame(self, frame_number):
        """
        Get the features for LSTM for a specific frame.
        """
        # Helper function to find frame data
        def find_frame_data(features_list):
            for features in features_list:
                if features['frame_number'] == frame_number:
                    return features
            return None

        return {
            'distances': find_frame_data(self.distances),
            'velocity': find_frame_data(self.velocity),
            'directions': find_frame_data(self.directions),
            'area_checks': find_frame_data(self.area_checks),
            'trajectory_analysis': find_frame_data(self.trajectory_analysis),
            'ball_locations': find_frame_data(self.ball_locations),
            'ball_player_features': find_frame_data(self.ball_player_features)
        }
    
    def get_ball_dataframe(self):
        """
        Converts ball information and related features into a pandas DataFrame.
        """
        all_frames_data = []
        
        # Iterate over all frame numbers available in ball positions
        for frame_data in self.ball_locations:
            frame_number = frame_data['frame_number']
            
            # Get all the relevant data for the current frame number
            distances = next((item for item in self.distances if item['frame_number'] == frame_number), {})
            velocity = next((item for item in self.velocity if item['frame_number'] == frame_number), {})
            directions = next((item for item in self.directions if item['frame_number'] == frame_number), {})
            area_checks = next((item for item in self.area_checks if item['frame_number'] == frame_number), {})
            trajectory_analysis = next((item for item in self.trajectory_analysis if item['frame_number'] == frame_number), {})
            #ball_player_features = next((item for item in self.ball_player_features if item['frame_number'] == frame_number), {})

            # Combine all these dictionaries into a single dictionary for this frame
            combined_data = {
                **frame_data,
                **distances,
                **velocity,
                **directions,
                **area_checks,
                **trajectory_analysis
                #**ball_player_features
            }
            
            all_frames_data.append(combined_data)

        # Convert the list of dictionaries to a DataFrame
        return pd.DataFrame(all_frames_data)

    
