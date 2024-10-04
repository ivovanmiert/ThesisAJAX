from feature_ball_related import BallInfo
from feature_player_related import PlayerInfo
import pandas as pd

class FeatureListGeneral:
    def __init__(self, ball, field_keypoints, players, max_players):
        """
        Initializes the FeatureListGeneral object with the given ball, field keypoints, and players.
        """
        self.field_keypoints = field_keypoints
        self.ball = ball
        self.players_info = self.initialize_players_info(players)
        self.ball_info = BallInfo(ball, field_keypoints, players)
        self.max_players = max_players

    def initialize_players_info(self, players):
        """
        Initializes PlayerInfo objects for each player in the Players object.
        """
        players_info = {}
        for player_id, player in players.players_dict.items():
            players_info[player_id] = PlayerInfo(player_id, players, self.ball, self.field_keypoints)
        return players_info

    def get_ball_features(self):
        """
        Retrieves all features from the BallInfo object.
        """
        return self.ball_info.get_features_for_lstm()

    def get_features_per_player(self, player_id):
        """
        Retrieves all features for a specific player.
        """
        player_info = self.players_info.get(player_id)
        all_features = player_info.get_all_features()
        return all_features
    
    
    def collect_all_features(self):
        """
        Collects all features from both the ball and the players.
        """
        players_df = self.get_all_player_dataframes()
        ball_df = self.get_dataframe_ball()
        # Merge ball and players DataFrames, assuming they are aligned on a common index or frame number
        if not players_df.empty:
            all_features_df = pd.merge(ball_df, players_df, how='outer', on='frame_number')
        else:
            all_features_df = ball_df
        
        return all_features_df
    
    def get_all_player_dataframes(self):
        all_dataframes = []

        # Iterate through each PlayerInfo object and get their features dataframe
        for player_id in self.players_info:
            player_info = self.players_info[player_id]
            df = player_info.get_all_features()
            all_dataframes.append(df)

        # Merge all dataframes into a single dataframe
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        sorted_df = combined_df.sort_values(by='frame_number').reset_index(drop=True)

        return sorted_df
    
    def get_dataframe_ball(self):
        df = self.ball_info.get_ball_dataframe()
        return df
