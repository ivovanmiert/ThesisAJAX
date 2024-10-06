import pickle
import numpy as np
import math

"""
This class contains a Players object, in which all Player objects are stored. It thus is a collection of Player objects. 
With the Players object, certain actions of merging Player objects for re-identification, or deleting Player objects when they are outside of the field boundaries can be done as well
"""


class Players:
    def __init__(self):
        """
        Initializes an empty dictionary to store Player objects.
        """
        self.players_dict = {}
        #self.hpe_matcher = hpe_matcher
    
    def add_player(self, player):
        """
        Adds a Player object to the dictionary.
        """
        self.players_dict[player.player_id] = player

    def update_player_position(self, player_id, current_frame_number, new_pixel_coordinates, new_pitch_coordinates, new_hpe_pixel_coordinates):
        """
        Updates the position of a specific Player object.
        """
        if player_id in self.players_dict:
            self.players_dict[player_id].update_position(current_frame_number, new_pixel_coordinates, new_pitch_coordinates, new_hpe_pixel_coordinates)
        else:
            raise ValueError(f"Player with ID {player_id} does not exist.")
        
    def update_player_hpe_assignment(self, player_id, frame_number, hpe_matcher):
        if player_id in self.players_dict:
            player = self.players_dict[player_id]
            pixel_bounding_box = player.pixel_coordinates

            #Find the next best HPE match for the player
            hpe_matches = hpe_matcher.match_single_detection(frame_number, pixel_bounding_box)
            for hpe_coordinates, hpe_id, current_iou in hpe_matches:
                if hpe_coordinates is not None:
                    assigned, assigned_iou = hpe_matcher.is_hpe_assigned(hpe_id)
                    if not assigned or current_iou > assigned_iou:
                        if assigned:
                            previous_player_id = hpe_matcher.matched_hpe[hpe_id]['player_id']
                            hpe_matcher.unassign_hpe(hpe_id)
                            self.update_player_hpe_assignment(previous_player_id, frame_number)
                        hpe_matcher.assign_hpe(hpe_id, player_id, current_iou)
                        player.hpe_coordinates = hpe_coordinates
                        return
            # When there is no new match found, set hpe_coordinates to None
            player.hpe_coordinates = None

    def get_player(self, player_id):
        """
        Retrieves a specific Player object.
        """
        return self.players_dict.get(player_id, None)

    
    def remove_player(self, player_id):
        if player_id in self.players_dict:
            del self.players_dict[player_id]
    
    def remove_nan_players(self):
        """
        Removes all Player objects with player_id equal to NaN from the dictionary.
        """
        players_to_remove = [player_id for player_id, player in self.players_dict.items() if self.is_nan(player.player_id)]
        for player_id in players_to_remove:
            self.remove_player(player_id)

    @staticmethod
    def is_nan(value):
        """
        Checks if a value is NaN
        """
        return isinstance(value, float) and math.isnan(value)

    def merge_players(self, frame_width=1920, frame_height=1080):
        def is_at_side(bbox, frame_width, frame_height, margin=50):
            x_min, y_min, x_max, y_max = bbox
            return (
                x_min < margin or x_max > frame_width - margin or
                y_min < margin or y_max > frame_height - margin
            )

        def should_merge(player1, player2, max_frame_diff=15, max_pitch_dist=5):
            if player1.team_classification != player2.team_classification:
                return False
            if player2.first_frame_number() - player1.current_frame_number > max_frame_diff or player2.first_frame_number() - player1.current_frame_number < 0:
                return False
            # if is_at_side(player1.last_known_location()[0], frame_width, frame_height) or is_at_side(player2.first_known_location()[0], frame_width, frame_height):
            #     return False
            pitch_dist = np.linalg.norm(np.array(player1.last_known_location()[1]) - np.array(player2.first_known_location()[1]))
            if pitch_dist > max_pitch_dist:
                return False
            return True

        players_list = list(self.players_dict.values())
        merged_checked_players = []

        for player in players_list:
            found = False
            for player_checked in merged_checked_players:
                if should_merge(player_checked, player):
                    player_checked.merge(player)
                    found = True
                    break
            if not found:
                merged_checked_players.append(player)

        # Create a new dictionary with merged players only
        self.players_dict = {player.player_id: player for player in merged_checked_players}

    def get_all_players(self):
        """
        Retrieves all Player objects.
        """
        return list(self.players_dict.values())
    
    def get_highest_player_id(self):
        """
        Retrieves the player ID of the player with the highest player ID.
        """
        if not self.players_dict:
            return None
        
        highest_id = max(self.players_dict.keys())
        return highest_id
    
    def remove_players_outside_field(self, field_keypoints):
        """
        Checks each player's positions across frames and removes players who are outside the field
        for more than 10% of the frames they have positions for.
        """
        players_to_remove = []

        for player_id, player in self.players_dict.items():
            total_positions = len(player.get_all_positions())  # Including current and previous positions
            outside_positions = 0

            # Check all positions (current + previous) to see if they are inside or outside the field
            for position in player.get_all_positions():
                pitch_coordinates = position['pitch_coordinates']
                
                # Check if the position is inside the defined area
                if not field_keypoints.is_position_inside_field(pitch_coordinates):
                    #print('deze hoort er niet in te zitten:')
                    #print(player_id, pitch_coordinates)
                    outside_positions += 1

            # Calculate the percentage of frames the player was outside the field
            outside_percentage = (outside_positions / total_positions) * 100

            # Mark the player for removal if they were outside the field for more than 10% of frames
            if outside_percentage > 10:
                players_to_remove.append(player_id)
        
        print(f"players to remove: {players_to_remove}")
        # Remove the marked players from the dictionary
        for player_id in players_to_remove:
            #print('player_id: going to check')
            #print(self.get_player(player_id))
            self.remove_player(player_id)

    def to_serializable_dict(self):
        """
        Converts the players_dict to a JSON-serializable format.
        """
        def convert_to_int(value):
            if isinstance(value, np.int64):
                return int(value)
            elif isinstance(value, dict):
                return {k: convert_to_int(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_to_int(i) for i in value]
            else:
                return value
        players_dictionary_form_dict = {
            int(player_id): convert_to_int(player.to_dict())
            for player_id, player in self.players_dict.items()
        }
        return players_dictionary_form_dict

    def __repr__(self):
        return f"Players({self.players_dict})"
    

