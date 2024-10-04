import json
import matplotlib.path as mpltPath
import math

"""
This file contains 2 classes:
    - Class 1: FieldKeypoints class --> which creates an object for one field type and processes the coordinates of it
    - Class 2: AllFieldKeypoints class --> which creates an object that collects all different FieldKeypoints object for each FieldKeypoint 

The goal of these classes is to prepare the 4 different kind of field types there are by the difference dimensions. The different field dimensions are:
    - 105 x 68m 
    - 105 x 67m
    - 110 x 68m
    - 105 x 65m

These different field dimensions are important during the field localization process. These different field dimensions result in different absolute places of the keypoints. 
These different places of the keypoints result in different places of the players on the field and different distances between players. 

The coordinates of the different keypoints can be found in the input json_file. These are manually calculated using the rules of football pitch dimensions by FIFA, pythogoras, and cosinus/sinus. 
"""


class FieldKeypoints:
    def __init__(self, keypoints):
        """
        Initializes the FieldKeypoints object with the given keypoints.

        """
        self.keypoints = keypoints  # A dictionary with keypoint names as keys and (x, y) tuples as values

    @staticmethod
    def from_json(json_file, field_type):
        """
        Loads keypoints from a JSON file for a specific field type and initializes the FieldKeypoints object.
        """
        with open(json_file, 'r') as file:
            data = json.load(file)

        keypoints = {}
        points = data.get(field_type, [])
        for i, point in enumerate(points):
            if point['x'] is not None and point['y'] is not None:
                keypoints[i] = (point['x'], point['y'])  # Using index as temporary names

        return FieldKeypoints(keypoints)
    
    @staticmethod
    def is_in_polygon(point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    @staticmethod
    def is_point_in_polygon(point, polygon):
        """
        Uses the matplotlib path to determine if a point is inside a polygon.
        """
        # Create a Path object from the polygon
        path = mpltPath.Path(polygon)
        # Check if the point is inside the polygon
        return path.contains_point(point)

    @staticmethod
    def is_point_in_polygon2(point, polygon):
        """
        Uses the ray-casting algorithm to determine if a point is inside a polygon.
        """
        x, y = point
        inside = False
        n = len(polygon)
        print(f"x : {x}")
        print(f"y : {y}")
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]  # Next vertex, wrapping around

            if (y1 > y) != (y2 > y):  # Check if point is between y-coordinates of the edge
                x_intersect = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x < x_intersect:
                    inside = not inside
        print(inside)
        return inside

    @staticmethod
    def calculate_distance(point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def get_keypoints(self):
        """
        Returns the keypoints stored in this FieldKeypoints object.
        """
        return self.keypoints
    
    def is_in_middle_third(self, point):
        max_x = max(kp[0] for kp in self.keypoints.values())
        third_x = max_x / 3
        if third_x <= point[0] <= 2 * third_x:
            return 1
        else: 
            return 0

    def is_in_left_penalty_area(self, point):
        kp = self.keypoints
        if self.is_in_polygon(point, [kp[11], kp[9], kp[8], kp[10]]):
            return 1
        else: 
            return 0

    def is_in_right_penalty_area(self, point):
        kp = self.keypoints
        if self.is_in_polygon(point, [kp[17], kp[19], kp[18], kp[16]]):
            return 1
        else: 
            return 0

    def is_in_left_deep_completion_area(self, point):
        mid_x = (self.keypoints[2][0] + self.keypoints[3][0]) / 2
        mid_y = (self.keypoints[2][1] + self.keypoints[3][1]) / 2
        if self.calculate_distance(point, (mid_x, mid_y)) <= 20:
            return 1
        else:
            return 0

    def is_in_right_deep_completion_area(self, point):
        mid_x = (self.keypoints[26][0] + self.keypoints[27][0]) / 2
        mid_y = (self.keypoints[26][1] + self.keypoints[27][1]) / 2
        if self.calculate_distance(point, (mid_x, mid_y)) <= 20:
            return 1
        else:
            return 0
        
    def is_position_inside_field(self, pitch_coordinates):
        """
        Checks if the given pitch coordinates are inside the polygon defined by keypoints 12, 13, 28, and 29.
        """
        # Coordinates of the four keypoints (12, 13, 28, 29) that form the field boundary
        polygon = [
            self.keypoints[12],  # Top-left corner
            self.keypoints[13],  # Bottom-left corner
            self.keypoints[28],  # Bottom-right corner
            self.keypoints[29]   # Top-right corner
        ]

        x, y = pitch_coordinates[0][0]
        if polygon[0][0] <= x and polygon[2][0] >= x and polygon[1][1] <= y and polygon[2][1] >= y:
            return True 
        else: 
            return False

class AllFieldKeypoints:
    def __init__(self, json_file):
        self.field_keypoints = {}
        self.load_keypoints(json_file)

    def load_keypoints(self, json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
        for field_type in data.keys():
            self.field_keypoints[field_type] = FieldKeypoints.from_json(json_file, field_type)

    def get_object_certain_type(self, field_type):
        return self.field_keypoints.get(field_type)
