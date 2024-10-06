This folder contains the code regarding the handling of the classification process. 

The code for the classification process itself can be found in /NEW_OBJECT/CLASSIFICATION_MODEL/ . Functions from that folder are imported in this folder. 

The 'job_run_classification.job' file was used to initially try out if the classification process worked. After this, the 'hyperparameter_search.sh' file was introduced with its corresponding 'job_run_classification_parameters.sh', which handles the hyperparameter searching process. Both jobs point towards the 'run_classification_balanced_different_cl.py' file which handles, based on the inputs in the configuration file and the parameters in the 'hyperparameter_search.sh' the right kind of classification, and defines the output paths for the results. 

The 'labels.py' file was used to run the creation of the labels for the events involved in the classification process. 
