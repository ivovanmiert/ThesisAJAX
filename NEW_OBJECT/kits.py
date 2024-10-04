import pandas as pd



"""
This class is for importing the different kits worn in each match and the field type there is played on for every match. 
This information is manually collected by watching the game footage and manually annotating which kit is worn. Since this was a process of annotating over 300 games, it is probable that a few mistakes are made. 
This information can be found in the input excel file. 
The goal of this class is to just import this information, so that it can later be easily extracted for every match separate if needed for a certain clip
Also a few errors in the dataset are handled, these errors were detected at a later stage in this project, so are set right here in a manual way. 


"""
class AllKits_Fieldtypes:
    def __init__(self, excel_file):
        self.kits_and_field_type = {}
        self.load_kits_and_field_type(excel_file)

    def load_kits_and_field_type(self, excel_file):
        df = pd.read_excel(excel_file)
        for _, row in df.iterrows():
            game_id = row['Game_ID']
            team_1 = row['Team 1']
            team_2 = row['Team 2']
            self.kits_and_field_type[game_id] = {
                'team_1': row['Team 1'],
                'team_2': row['Team 2'],
                'kit_team_1': f"{team_1}_{row['Kit Team 1']}",
                'kit_team_2': f"{team_2}_{row['Kit Team 2']}",
                'keeper_kit_team_1': f"{team_1}_Keeper{row['Keeper Kit Team 1']}",
                'keeper_kit_team_2': f"{team_2}_Keeper{row['Keeper Kit Team 2']}",
                'referee_kit': f"referee_{row['Referee Kit']}",
                'field_type': row['Field_Type']
            }
            #Now some changes/errors in class names which are treated right here. First error is that a class is called lecceKeeper4 instead of lecce_Keeper4. The other Error is that Bologna_Keeper1 is 2 times as a kit in the dataset, but this needs to be Bologna_Keeper4.
            if team_1== 'lecce' and row['Keeper Kit Team 1'] == 4:
                self.kits_and_field_type[game_id] = {
                'team_1': row['Team 1'],
                'team_2': row['Team 2'],
                'kit_team_1': f"{team_1}_{row['Kit Team 1']}",
                'kit_team_2': f"{team_2}_{row['Kit Team 2']}",
                'keeper_kit_team_1': f"{team_1}Keeper{row['Keeper Kit Team 1']}",
                'keeper_kit_team_2': f"{team_2}_Keeper{row['Keeper Kit Team 2']}",
                'referee_kit': f"referee_{row['Referee Kit']}",
                'field_type': row['Field_Type']
            }
            if team_2 == 'lecce' and row['Keeper Kit Team 2'] == 4:
                self.kits_and_field_type[game_id] = {
                'team_1': row['Team 1'],
                'team_2': row['Team 2'],
                'kit_team_1': f"{team_1}_{row['Kit Team 1']}",
                'kit_team_2': f"{team_2}_{row['Kit Team 2']}",
                'keeper_kit_team_1': f"{team_1}_Keeper{row['Keeper Kit Team 1']}",
                'keeper_kit_team_2': f"{team_2}Keeper{row['Keeper Kit Team 2']}",
                'referee_kit': f"referee_{row['Referee Kit']}",
                'field_type': row['Field_Type']
            }
            if team_1 =='bologna' and row['Keeper Kit Team 1'] == 1:
                self.kits_and_field_type[game_id] = {
                'team_1': row['Team 1'],
                'team_2': row['Team 2'],
                'kit_team_1': f"{team_1}_{row['Kit Team 1']}",
                'kit_team_2': f"{team_2}_{row['Kit Team 2']}",
                'keeper_kit_team_1': f"{team_1}_Keeper4",
                'keeper_kit_team_2': f"{team_2}_Keeper{row['Keeper Kit Team 2']}",
                'referee_kit': f"referee_{row['Referee Kit']}",
                'field_type': row['Field_Type']
            }
            if team_2 =='bologna' and row['Keeper Kit Team 2'] == 1:
                self.kits_and_field_type[game_id] = {
                'team_1': row['Team 1'],
                'team_2': row['Team 2'],
                'kit_team_1': f"{team_1}_{row['Kit Team 1']}",
                'kit_team_2': f"{team_2}_{row['Kit Team 2']}",
                'keeper_kit_team_1': f"{team_1}_Keeper{row['Keeper Kit Team 1']}",
                'keeper_kit_team_2': f"{team_2}_Keeper4",
                'referee_kit': f"referee_{row['Referee Kit']}",
                'field_type': row['Field_Type']
            }
                
    def get_kits_and_field_type_whole(self):
        return self.kits_and_field_type


    def get_kits_and_field_type(self, game_id):
        return self.kits_and_field_type.get(game_id)