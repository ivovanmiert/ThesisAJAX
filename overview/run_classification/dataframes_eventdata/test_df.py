import pandas as pd

df = pd.read_csv('/home/ivanmiert/overview/FRAMES_EXTRACTION/dataframes_eventdata/all_selected.csv')

print(df['pass_accurate'].unique())
print(df[df['on_target'] == 0]['primary_type']) #['back_pass'].value_counts())
#print(df[df['pass'] == True]['accurate'])