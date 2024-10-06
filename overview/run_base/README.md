This folder contains the two job scripts 'job_base.job' and 'job_concatenate_dataframes.job'

'job_base.job':
This job script runs the base.py script. In this script the right input and output paths are defined, as well as the part of the data being processed defined by the parameters 'current_chunk' and 'part'. (Current chunk selects 1 of the 8 chunks of 1000 events, then 'part' selects a part of that data, defined in 'check_intersection.py'. This way, the data could be splitted into smaller parts, and ran in parallel. Then after this, the data is processed by instantiating the Events class from the 'load_and_work_overview' file found in the 'NEW_OBJECT' folder. 

'job_concatenate_dataframes.job':
This job script runs the concatenate_dataframes.py script. This file import the 'combine_player_ball_data' function from the 'example_overview' file. This file can be found in the NEW_OBJECT/CLASSIFICATION_MODEL/ folder. This function is regarding the concatenation of the features for ball and player. These features were saved seperately after running the 'job_base.job' and were needed to be concatenated for the classification process. This concatenation process can be found in the 'example_overview' file. 
