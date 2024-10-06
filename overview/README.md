This folder contains three folders and a configuration file (config.yaml). From this folder, jobs are sent to the Snellius HPC.
The three different folders are:
  - run_models: In this folder, each different model has its own folder in which can be found for every model its job files and python script that imports the 'main' function of that model. This python scripts handles the processing of the data.
  - run_base: This folder handles the processing of the data into features.
  - run_classification: This folder handles the classification process.

Besides the three folders. This folder contains a configuration file. This configuration file is used to save the paths of different locations and make it easy to access them. Besides this, it controls certain parameters that control which functions are ran with what parameters. Since the processing of images and videos used a lot of computation power, most methods are used in parallel on different parts of the data. This is for example controlled by the variables 'chunk' and 'part'. Which can be found in most python files ran in the job scripts. 
