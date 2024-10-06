# ThesisAJAX
Github Repository for my Thesis Internship at AFC AJAX


This repository contains the code for my thesis internship
The repository consits of different folders, all regarding a different part in the total thesis. It contains the following main folders:
  - MODELS_USED: This folder contains all existing models used/implemented. This is for the models of field localization (field_localization), player detection and tracking (tracking), ball detection (football), the top-down HPE model (pose-estimation) and the bottom-up HPE model (pifpaf). This folder also contains the self-designed/written model for Team Classification
  - overview: This folder contains all scripts regarding the running of the different models. This folder can be seen as the center of the project in terms of handling jobs. It is the communicating heart of the project. From this folder, the different models are imported and used on the data. It is further described in its README file. 
  - NEW_OBJECT: This folder contains all the code regarding the processing of the data obtained from the different models mentioned in 'MODELS_USED' and creating the features that serve as input to the classification system. Next to this, 'NEW_OBJECT' contains the folder 'CLASSIFICATION_MODEL', which contains the code on creating, training, validating and evaluating the classification models. 
  - environments: This folder contains all different requirements files for the different environments. These environments need to be installed in order to be able to run the different models/jobs on Snellius.
  - DATA CEP: This folder contains files regarding the data collection, data exploration and data preprocessing

Because of the high computation power needed to process images/videos, all jobs/code was ran on HPC system Snellius. Snellius was accessed by sending jobs with the scripts to it. These jobs then entered a queue and were processed as soon as there were computing nodes available at Snellius. Different .job files can be found in the different repositories (mainly in overview). These job files show a request of a job to Snellius. In the upper part of the job it shows the different resources asked for, followed by the loading of different modules and environment. From there a script is ran.  

