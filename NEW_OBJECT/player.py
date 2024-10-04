
class Player:
    def __init__(self, player_id, team_classification, pixel_coordinates, pitch_coordinates, hpe_pixel_coordinates, current_frame_number, hpe_only):
        """
        Initializes a new player with the provided attributes.
        """
        self.player_id = player_id
        self.team_classification = team_classification
        self.pixel_coordinates = pixel_coordinates
        self.pitch_coordinates = pitch_coordinates
        self.hpe_pixel_coordinates = hpe_pixel_coordinates
        self.current_frame_number = current_frame_number #will most of the time be 0, but sometimes it won't. 
        self.previous_positions= []
        self.hpe_only = hpe_only
       
    def update_position(self, frame_number, new_pixel_coordinates, new_pitch_coordinates, new_hpe_pixel_coordinates):
        """
        Updates the player's position and stores the previous positions with frame numbers.
        """            
        #Store the current positions with the frame number in previous_positions
        self.previous_positions.append({
            'frame_number': self.current_frame_number,
            'pixel_coordinates': self.pixel_coordinates,
            'pitch_coordinates': self.pitch_coordinates,
            'hpe_pixel_coordinates': self.hpe_pixel_coordinates
        })
        #Update to new positions
        self.current_frame_number = frame_number
        self.pixel_coordinates = new_pixel_coordinates
        self.pitch_coordinates = new_pitch_coordinates
        self.hpe_pixel_coordinates = new_hpe_pixel_coordinates
    
    def get_previous_positions(self):
        """
        Returns the previous positions with their corresponding frame numbers.
        """
        return self.previous_positions
    
    def get_all_positions(self):
        """
        Returns all positions including the current one.
        """
        all_positions = self.previous_positions.copy()
        all_positions.append({
            'frame_number': self.current_frame_number,
            'pixel_coordinates': self.pixel_coordinates,
            'pitch_coordinates': self.pitch_coordinates,
            'hpe_pixel_coordinates': self.hpe_pixel_coordinates
        })
        return all_positions
    
    def last_known_location(self):
        return self.pixel_coordinates, self.pitch_coordinates, self.hpe_pixel_coordinates

    def first_known_location(self):
        if self.previous_positions:
            first_pos = self.previous_positions[0]
            return first_pos['pixel_coordinates'], first_pos['pitch_coordinates'], first_pos['hpe_pixel_coordinates']
        return self.pixel_coordinates, self.pitch_coordinates, self.hpe_pixel_coordinates
    
    def merge(self, other):
        self.previous_positions.append({
            'frame_number': self.current_frame_number,
            'pixel_coordinates': self.pixel_coordinates,
            'pitch_coordinates': self.pitch_coordinates,
            'hpe_pixel_coordinates': self.hpe_pixel_coordinates
        })
        self.previous_positions.extend(other.previous_positions)
        self.update_position(
            other.current_frame_number,
            other.pixel_coordinates,
            other.pitch_coordinates,
            other.hpe_pixel_coordinates
        )
    
    def first_frame_number(self):
        if self.previous_positions:
            return self.previous_positions[0]['frame_number']
        return self.current_frame_number
        
    def get_position_at_frame(self, frame_number):
        """
        Gets the player's position at a specific frame number.
        """
        for position in self.previous_positions:
            if position['frame_number'] == frame_number:
                return position
        # Check the current position if the frame number matches
        if self.current_frame_number == frame_number:
            return {
                'frame_number': self.current_frame_number,
                'pixel_coordinates': self.pixel_coordinates,
                'pitch_coordinates': self.pitch_coordinates,
                'hpe_pixel_coordinates': self.hpe_pixel_coordinates
            }
        # Return None if the frame number is not found
        return None
    
    def to_dict(self):
        return {
            'player_id': self.player_id,
            'pixel_coordinates': self.pixel_coordinates,
            'pitch_coordinates': self.pitch_coordinates,
            'current_frame_number': self.current_frame_number,
            'previous_positions': self.previous_positions,
        }

    
    def __repr__(self):
        return (f"Player(player_id={self.player_id}, "
                f"team_classification={self.team_classification}, "
                f"pixel_coordinates={self.pixel_coordinates}, "
                f"pitch_coordinates={self.pitch_coordinates}, "
                f"hpe_pixel_coordinates={self.hpe_pixel_coordinates}, "
                f"previous_positions={self.previous_positions})")




#These two functions are used to retrieve the player object from a file and to save a player object to a file respectively:

def get_player_from_file(file_path):
    """
    Reads a player object from a file.
    """
    import pickle

    with open(file_path, 'rb') as file:
        player = pickle.load(file)
    
    return player

def save_player_to_file(player, file_path):
    """
    Saves a player object to a file.
    """
    import pickle

    with open(file_path, 'wb') as file:
        pickle.dump(player, file)