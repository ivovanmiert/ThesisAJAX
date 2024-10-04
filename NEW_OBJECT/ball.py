import numpy as np


"""
This class handles the loading of the data concerning the ball. 
"""

class Ball:
    def __init__(self, pixel_coordinates, pitch_coordinates, current_frame_number):
        """
        Initializes a new ball with the provided attributes.
        """
        self.pixel_coordinates = pixel_coordinates
        self.pitch_coordinates = pitch_coordinates
        self.previous_positions = []
        self.current_frame_number = current_frame_number

    def update_position(self, frame_number, new_pixel_coordinates, new_pitch_coordinates, overwrite=False):
        """
        Updates the ball's position and stores the previous positions with frame numbers.
        """
        if not overwrite:
            # Store the current positions with the current frame number in previous_positions
            self.previous_positions.append({
                'frame_number': self.current_frame_number,
                'pixel_coordinates': self.pixel_coordinates,
                'pitch_coordinates': self.pitch_coordinates,
                'interpolated': False
            })
            
        # Update to new positions
        self.current_frame_number = frame_number
        self.pixel_coordinates = new_pixel_coordinates
        self.pitch_coordinates = new_pitch_coordinates

    def get_previous_detection(self):
        """
        Returns the pixel coordinates of the last detection of the ball.
        """
        if self.previous_positions:
            return self.previous_positions[-1]['pixel_coordinates']
        return self.pixel_coordinates

    def __repr__(self):
        return (f"Ball(pixel_coordinates={self.pixel_coordinates}, "
                f"pitch_coordinates={self.pitch_coordinates}, "
                f"previous_positions={self.previous_positions})")
    
    def interpolate_positions(self):
        """
        Interpolates positions for frames with no detections between known detections.
        """
        if not self.previous_positions:
            return []
        
        all_positions = self.previous_positions + [{
            'frame_number': self.current_frame_number,
            'pixel_coordinates': self.pixel_coordinates,
            'pitch_coordinates': self.pitch_coordinates,
            'interpolated': False
        }]
        
        all_positions.sort(key=lambda x: x['frame_number'])
        interpolated_positions = []
        
        for i in range(len(all_positions) - 1):
            start = all_positions[i]
            end = all_positions[i + 1]
            frame_diff = end['frame_number'] - start['frame_number']
            
            if frame_diff > 1:
                print('bij deze:')
                print(i)
                for frame in range(1, frame_diff):
                    ratio = frame / frame_diff
                    interp_pixel_x = start['pixel_coordinates'][0] + ratio * (end['pixel_coordinates'][0] - start['pixel_coordinates'][0])
                    interp_pixel_y = start['pixel_coordinates'][1] + ratio * (end['pixel_coordinates'][1] - start['pixel_coordinates'][1])
                    start_pitch = start['pitch_coordinates'][0, 0]
                    end_pitch = end['pitch_coordinates'][0, 0]
                    interp_pitch_x = start_pitch[0] + ratio * (end_pitch[0] - start_pitch[0])
                    interp_pitch_y = start_pitch[1] + ratio * (end_pitch[1] - start_pitch[1])
                    interp_pitch_coordinates = np.array([[[interp_pitch_x, interp_pitch_y]]])
                    
                    interpolated_positions.append({
                        'frame_number': start['frame_number'] + frame,
                        'pixel_coordinates': (interp_pixel_x, interp_pixel_y),
                        'pitch_coordinates': interp_pitch_coordinates, 
                        'interpolated': True
                    })
        self.previous_positions.extend(interpolated_positions)
        self.previous_positions.sort(key=lambda x: x['frame_number'])
        
        return interpolated_positions
    
    def get_all_positions(self):
        """
        Returns all positions, including interpolated ones, sorted by frame number.
        """
        all_positions = self.previous_positions + [{
            'frame_number': self.current_frame_number,
            'pixel_coordinates': self.pixel_coordinates,
            'pitch_coordinates': self.pitch_coordinates,
            'interpolated': False
        }]
        #print(f"ball all positions: {all_positions}")
        all_positions.sort(key=lambda x: x['frame_number'])
        
        # Ensure every position has the 'interpolated' attribute
        for pos in all_positions:
            if 'interpolated' not in pos:
                pos['interpolated'] = False
        
        return all_positions
